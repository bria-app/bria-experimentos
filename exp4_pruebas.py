"""Experimento 4 - Suite de pruebas del motor y latencia de una solicitud de un producto.

1. Ejecuta ``python -m pytest -q`` dentro de inventory-dss-ftgm-engine (PYTHONPATH = motor)
   y guarda la salida literal en logs/exp4_pytest.txt.
2. Mide la latencia de ForecastService().forecast() para UN producto (contrato de la API:
   frequency="auto", horizon_days=30 y 90, as_of=2016-02-01, puntos diarios dispersos) con
   historias de 12 y 24 meses de un producto M5 fijo (el primero de la muestra del Exp. 3).
   50 repeticiones medidas tras 3 de calentamiento; se reportan mediana, p90, min y max.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import common as C


def main() -> None:
    sys.stdout = C.Tee(C.LOGS / "exp4_log.txt")  # type: ignore[assignment]
    print("EXPERIMENTO 4 - Pruebas automatizadas y latencia")
    print(C.environment())

    env = dict(os.environ, PYTHONPATH=str(C.ENGINE))
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=C.ENGINE, env=env,
                          capture_output=True, text=True)
    dt = time.time() - t0
    out = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")
    (C.LOGS / "exp4_pytest.txt").write_text(
        f"$ cd {C.ENGINE}\n$ PYTHONPATH={C.ENGINE} {sys.executable} -m pytest -q\n"
        f"(codigo de salida {proc.returncode}, {dt:.1f} s)\n\n{out}", encoding="utf-8")
    print(f"pytest: codigo de salida {proc.returncode} en {dt:.1f} s; ultima linea: "
          f"{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else 'n/d'}")

    import exp3_series_cortas as E3

    sample, S = E3.sample_products()
    full = S[0]
    rows = []
    for L in (12, 24):
        start = C.add_months(E3.AS_OF, -L)
        pts = E3.make_points(full[E3.day_index(start) : E3.day_index(E3.AS_OF)], start)
        for H in (30, 90):
            for _ in range(3):
                E3.run_engine(pts, H)
            times = []
            for _ in range(50):
                t = time.perf_counter()
                fc = E3.run_engine(pts, H)
                times.append(time.perf_counter() - t)
            a = np.array(times) * 1000
            rows.append({"producto": sample.loc[0, "id"], "L_meses": L, "H_dias": H, "frecuencia": fc.frequency,
                         "modelo_ganador": fc.model, "n_puntos": len(pts), "repeticiones": a.size,
                         "mediana_ms": round(float(np.median(a)), 1), "p90_ms": round(float(np.percentile(a, 90)), 1),
                         "min_ms": round(float(a.min()), 1), "max_ms": round(float(a.max()), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "exp4_latencia.csv", index=False)
    print("\nLatencia de ForecastService para un producto (50 repeticiones, proceso unico):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
