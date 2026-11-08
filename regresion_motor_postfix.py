"""Regresion numerica del motor: commit be6a98f (tesis) vs. arbol con las correcciones de auditoria.

Uso:
    python regresion_motor_postfix.py <motor_antes> <motor_despues> <log.txt>

Cada motor se ejecuta en su propio subproceso (sys.path apunta a su carpeta), sobre las
mismas series del M5 (cache/m5_sales.npy) y los mismos contratos que usan la API y los
experimentos:

  * auto + horizon_days (30 y 90) + as_of = 2016-02-01, historias de 6, 12, 24 y 36 meses,
    puntos diarios dispersos (solo dias con venta), como exp3;
  * legado period = 12, horizon = 6, puntos mensuales preagregados (36 meses);
  * frecuencia semanal forzada (12 meses);
  * censura marcada (stockout_flag) en ventanas de 7 a 14 dias (24 meses), como exp3-C;
  * un caso sintetico "siempre agotado" (unica diferencia esperada, hallazgo M-15).

Se compara la respuesta completa (puntos, intervalos, historia, metricas, diagnosticos y
textos) con tolerancia 1e-9.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
M5_START = date(2011, 1, 29)
AS_OF = date(2016, 2, 1)
N_SERIES = 36
SEED = 42
TOL = 1e-9


def _add_months(d: date, k: int) -> date:
    m = d.month - 1 + k
    return date(d.year + m // 12, m % 12 + 1, 1)


def _idx(d: date) -> int:
    return (d - M5_START).days


def _cases() -> list[tuple[str, dict[str, Any]]]:
    import numpy as np
    import pandas as pd

    meta = pd.read_csv(CACHE / "m5_meta.csv")
    sales = np.load(CACHE / "m5_sales.npy")
    i_asof, i36 = _idx(AS_OF), _idx(_add_months(AS_OF, -36))
    mean24 = sales[:, _idx(_add_months(AS_OF, -24)) : i_asof].mean(axis=1)
    first = np.argmax(sales > 0, axis=1)
    rng = np.random.default_rng(SEED)
    regular = np.flatnonzero((mean24 >= 0.5) & (first <= i36))
    sparse = np.flatnonzero((mean24 > 0.02) & (mean24 < 0.2) & (first <= i36))  # intermitentes
    rows = sorted(rng.choice(regular, 30, replace=False).tolist() + rng.choice(sparse, 6, replace=False).tolist())

    def daily(row: int, months: int, flags: Any = None) -> list[dict[str, Any]]:
        start = _add_months(AS_OF, -months)
        hist = sales[row, _idx(start) : i_asof]
        out = []
        for k, v in enumerate(hist):
            f = bool(flags[k]) if flags is not None else False
            if v > 0 or f:
                out.append({"date": (start + timedelta(days=k)).isoformat(),
                            "demand": 0.0 if f else float(v), "stockout_flag": f})
        return out

    cases: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        pid = str(meta.loc[row, "id"])
        for months in (6, 12, 24, 36):
            pts = daily(row, months)
            for h in (30, 90):
                cases.append((f"{pid}|auto|L{months}|H{h}", {
                    "frequency": "auto", "horizon_days": h, "as_of": AS_OF.isoformat(),
                    "series": [{"product_id": pid, "points": pts}],
                }))
        monthly = []
        for m in range(36, 0, -1):
            s, e = _add_months(AS_OF, -m), _add_months(AS_OF, -m + 1)
            monthly.append({"date": s.isoformat(), "demand": float(sales[row, _idx(s) : _idx(e)].sum())})
        cases.append((f"{pid}|legacy|P12|H6", {"model": "FTGM", "period": 12, "horizon": 6,
                                               "series": [{"product_id": pid, "points": monthly}]}))
        cases.append((f"{pid}|weekly|L12|H90", {
            "frequency": "weekly", "horizon_days": 90, "as_of": AS_OF.isoformat(),
            "series": [{"product_id": pid, "points": daily(row, 12)}],
        }))
        crng = np.random.default_rng(SEED + row)
        n_days = _idx(AS_OF) - _idx(_add_months(AS_OF, -24))
        mask = np.zeros(n_days, dtype=bool)
        for _ in range(6):
            ln = int(crng.integers(7, 15))
            s0 = int(crng.integers(0, n_days - ln))
            mask[s0 : s0 + ln] = True
        cases.append((f"{pid}|censored|L24|H90", {
            "frequency": "auto", "horizon_days": 90, "as_of": AS_OF.isoformat(),
            "series": [{"product_id": pid, "points": daily(row, 24, mask)}],
        }))
    always_out = [{"date": (date(2015, 9, 1) + timedelta(days=k)).isoformat(), "demand": 0.0,
                   "stockout_flag": True} for k in range(_idx(AS_OF) - _idx(date(2015, 9, 1)))]
    cases.append(("SINTETICO_siempre_agotado|auto|H30", {
        "frequency": "auto", "horizon_days": 30, "as_of": AS_OF.isoformat(),
        "series": [{"product_id": "siempre-agotado", "points": always_out}],
    }))
    return cases


def dump(engine: str, out: str) -> None:
    sys.path.insert(0, engine)
    warnings.filterwarnings("ignore")
    from app.application.forecast_service import ForecastService
    from app.presentation.schemas import ForecastRequest

    svc = ForecastService()
    result: dict[str, Any] = {}
    t0 = time.perf_counter()
    for key, body in _cases():
        resp = svc.forecast(ForecastRequest.model_validate(body))
        result[key] = resp.model_dump(mode="json")
    result["__meta__"] = {"engine": engine, "seconds": round(time.perf_counter() - t0, 1)}
    Path(out).write_text(json.dumps(result), encoding="utf-8")


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: solo en {'despues' if k in b else 'antes'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: largo {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) or not math.isclose(
            a, b, rel_tol=TOL, abs_tol=TOL
        ):
            out.append(f"{path}: {a!r} -> {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}")


def main(before: str, after: str, log: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    tmp = Path(log).with_suffix("")
    files = {}
    for tag, engine in (("antes", before), ("despues", after)):
        f = Path(f"{tmp}_{tag}.json")
        subprocess.run([sys.executable, __file__, "--dump", engine, str(f)], check=True)
        files[tag] = json.loads(f.read_text(encoding="utf-8"))
        f.unlink()
    a, b = files["antes"], files["despues"]
    lines = [
        "Regresion numerica del motor FTGM: be6a98f (antes) vs. correcciones de auditoria (despues)",
        f"antes:   {before} ({a['__meta__']['seconds']} s)",
        f"despues: {after} ({b['__meta__']['seconds']} s)",
        f"tolerancia relativa/absoluta: {TOL}",
        "",
    ]
    keys = [k for k in a if k != "__meta__"]
    same, changed = 0, []
    for k in keys:
        d: list[str] = []
        _diff(a[k], b.get(k), k, d)
        if d:
            changed.append((k, d))
        else:
            same += 1
    statuses: dict[str, int] = {}
    models: dict[str, int] = {}
    for k in keys:
        for fc in a[k]["forecasts"]:
            statuses[fc["status"]] = statuses.get(fc["status"], 0) + 1
            models[fc["model"]] = models.get(fc["model"], 0) + 1
    n_series = len({k.split("|")[0] for k in keys})
    lines.append(f"Series: {n_series}; casos (series x contrato): {len(keys)}")
    lines.append(f"Estados (antes): {statuses}")
    lines.append(f"Modelos ganadores (antes): {models}")
    lines.append(f"Casos identicos: {same}; casos con diferencias: {len(changed)}")
    for k, d in changed:
        lines.append(f"\n-- {k} ({len(d)} diferencias)")
        lines.extend(f"   {x}" for x in d[:15])
        if len(d) > 15:
            lines.append(f"   ... {len(d) - 15} mas")
    expected = all(k.startswith("SINTETICO_siempre_agotado") for k, _ in changed)
    lines.append("")
    lines.append(
        "RESULTADO: OK - salidas identicas salvo el caso documentado 'siempre agotado' (M-15)"
        if expected
        else "RESULTADO: FALLA - hay diferencias en series normales"
    )
    Path(log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if expected else 1)


if __name__ == "__main__":
    if sys.argv[1] == "--dump":
        dump(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
