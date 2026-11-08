"""Experimento 7 - Simulacion de la decision de reposicion (RQ3 del paper JISEBI).

Pregunta: con la MISMA regla de pedido del sistema, ¿que pronostico deja mejores decisiones
de inventario (nivel de servicio, ventas perdidas, inventario promedio)?

Datos: los mismos 300 productos-tienda del M5 del Exp. 3 (muestra estratificada, semilla 42).

Protocolo (revision periodica mensual con ventas perdidas):
  * Seis revisiones el primer dia de cada mes, del 2015-09-01 al 2016-02-01; la simulacion
    cubre del 2015-09-01 al 2016-02-29 (182 dias).
  * Historia inicial de L = 6, 9, 12, 18 o 24 meses antes del 2015-09-01 que CRECE con cada
    revision, como la de una MYPE que sigue registrando ventas.
  * En cada revision se llama al servicio de produccion del motor
    (ForecastService().forecast con frequency="auto", horizon_days=90, as_of=revision).
    Las referencias (Naive y SNaive) se calculan sobre los mismos periodos y la misma historia
    cruda; el Oraculo usa la demanda real futura (cota superior de la politica).
  * Regla de la capa de decision del sistema (ecuacion 7 del paper): nivel objetivo
    O = D(L+R) con plazo de entrega L = 7 dias y revision R = 30 dias, donde D(w) reparte cada
    pronostico por periodo uniformemente en sus dias. Pedido Q = ceil(k*O - posicion), con k
    un factor de inventario de seguridad (k = 1 es la regla sin stock de seguridad).
  * Dinamica diaria: llega el pedido (7 dias despues de la revision), se atiende la demanda
    real con el stock disponible y lo no atendido se pierde. Stock inicial = ceil(k*O) de la
    primera revision.
  * Metricas: tasa de servicio (fill rate = atendido / demanda), ventas perdidas y dias de
    inventario (stock promedio / demanda diaria promedio).
  * Pruebas de Wilcoxon pareadas por producto con correccion de Holm.

Uso: ENGINE_DIR=<copia limpia del motor> ENGINE_COMMIT=<hash> python exp7_simulacion_decision.py
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
from datetime import date, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

_ENGINE_DIR = os.environ.get("ENGINE_DIR")
if _ENGINE_DIR:
    sys.path.insert(0, _ENGINE_DIR)

import exp3_series_cortas as E3  # noqa: E402  (reutiliza muestra y utilidades)

ORIGINS = [date(2015, 9, 1), date(2015, 10, 1), date(2015, 11, 1), date(2015, 12, 1),
           date(2016, 1, 1), date(2016, 2, 1)]
SIM_END = date(2016, 3, 1)  # exclusivo
LENGTHS = (6, 9, 12, 18, 24)
LEAD, REVIEW, HORIZON = 7, 30, 90
KS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
METHODS = ("Motor", "Naive", "SNaive", "Oraculo")


def di(d: date) -> int:
    return (d - C.M5_START).days


def run_engine(points: list, as_of: date):
    from app.application.forecast_service import ForecastService
    from app.presentation.schemas import ForecastRequest, ProductSeries

    req = ForecastRequest(frequency="auto", horizon_days=HORIZON, as_of=as_of,
                          series=[ProductSeries(product_id="p", points=points)])
    return ForecastService().forecast(req).forecasts[0]


def baselines(points: list, period: int, h: int, as_of: date) -> dict[str, np.ndarray]:
    from app.baselines import seasonal_naive_forecast
    from app.ftgm import preprocessing as prep

    b = prep.bucketize([p.date for p in points], np.array([p.demand for p in points]),
                       np.zeros(len(points), dtype=bool), period, as_of)
    x = b.demand
    return {"Naive": np.full(h, x[-1]), "SNaive": seasonal_naive_forecast(x, h, period)}


def demand_window(starts: list[date], values: np.ndarray, period: int, origin: date) -> float:
    """D(L+R): suma del pronostico repartido por dia en [origin, origin + L + R)."""
    from app.ftgm import preprocessing as prep

    end = origin + timedelta(LEAD + REVIEW)
    total, covered_to = 0.0, origin
    for s, v in zip(starts, values):
        e = prep.next_bucket(s, period)
        rate = max(float(v), 0.0) / (e - s).days
        a, b = max(s, origin), min(e, end)
        if b > a:
            total += rate * (b - a).days
            covered_to = max(covered_to, b)
    if covered_to < end and len(values):  # horizonte corto: se extiende el ultimo periodo
        s, v = starts[-1], values[-1]
        rate = max(float(v), 0.0) / (prep.next_bucket(s, period) - s).days
        total += rate * (end - covered_to).days
    return total


def simulate(daily: np.ndarray, targets: list[float], k: float) -> dict[str, float]:
    """Revision periodica con ventas perdidas. ``daily`` cubre [ORIGINS[0], SIM_END)."""
    review_idx = [(o - ORIGINS[0]).days for o in ORIGINS]
    on_hand = float(math.ceil(k * targets[0]))
    pipeline: dict[int, float] = {}
    served = lost = inv = 0.0
    orders = 0
    for t, d in enumerate(daily):
        if t in review_idx and t > 0:
            r = review_idx.index(t)
            position = on_hand + sum(pipeline.values())
            q = math.ceil(k * targets[r] - position)
            if q > 0:
                pipeline[t + LEAD] = pipeline.get(t + LEAD, 0.0) + q
                orders += 1
        on_hand += pipeline.pop(t, 0.0)
        s = min(on_hand, d)
        served += s
        lost += d - s
        on_hand -= s
        inv += on_hand
    dem = float(daily.sum())
    return {"demanda": dem, "atendido": served, "perdido": lost, "inv_unid_dia": inv, "pedidos": orders}


def work(args: tuple) -> tuple[list[dict], list[dict]]:
    warnings.filterwarnings("ignore")
    pid, item, cat, full = args
    sim_rows, tgt_rows = [], []
    daily = full[di(ORIGINS[0]) : di(SIM_END)].astype(float)
    for L in LENGTHS:
        start = C.add_months(ORIGINS[0], -L)
        targets = {m: [] for m in METHODS}
        for o in ORIGINS:
            pts = E3.make_points(full[di(start) : di(o)], start)
            fc = run_engine(pts, o)
            oracle = float(full[di(o) : di(o) + LEAD + REVIEW].sum())
            if fc.status == "skipped" or not fc.points:
                # El sistema no recomienda: se usa demanda cero para el motor.
                vals = {"Motor": 0.0, "Naive": 0.0, "SNaive": 0.0}
                status, model, period = fc.status, fc.model, None
            else:
                starts = [p.date for p in fc.points]
                f = np.array([p.predicted_demand for p in fc.points])
                base = baselines(pts, fc.period, len(starts), o)
                vals = {"Motor": demand_window(starts, f, fc.period, o)}
                vals |= {m: demand_window(starts, v, fc.period, o) for m, v in base.items()}
                status, model, period = fc.status, fc.model, fc.period
            vals["Oraculo"] = oracle
            for m in METHODS:
                targets[m].append(vals[m])
            tgt_rows.append({"producto": pid, "L_meses": L, "revision": o.isoformat(), "status": status,
                             "modelo_motor": model, "periodo": period, "demanda_real_37d": oracle,
                             **{f"D37_{m}": vals[m] for m in METHODS}})
        for m in METHODS:
            for k in KS:
                r = simulate(daily, targets[m], k)
                sim_rows.append({"producto": pid, "item": item, "cat": cat, "L_meses": L, "metodo": m, "k": k, **r})
    return sim_rows, tgt_rows


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def frontier(g: pd.DataFrame, target: float) -> float:
    """Dias de inventario necesarios para una tasa de servicio agregada ``target`` (interpolado en k)."""
    g = g.sort_values("k")
    fr, dd = g["fill"].to_numpy(), g["dias_inv"].to_numpy()
    for i in range(len(fr) - 1):
        if fr[i] <= target <= fr[i + 1] and fr[i + 1] > fr[i]:
            w = (target - fr[i]) / (fr[i + 1] - fr[i])
            return float(dd[i] + w * (dd[i + 1] - dd[i]))
    return float("nan")


def main() -> None:
    sys.stdout = C.Tee(C.LOGS / "exp7_log.txt")  # type: ignore[assignment]
    t0 = time.time()
    print("EXPERIMENTO 7 - Simulacion de la decision de reposicion")
    print(C.environment())
    print(f"Motor: {os.environ.get('ENGINE_DIR', str(C.ENGINE))} commit {os.environ.get('ENGINE_COMMIT', '?')}")
    print(f"Revisiones: {[o.isoformat() for o in ORIGINS]}; fin {SIM_END}; L={LENGTHS}; plazo {LEAD} d; "
          f"revision {REVIEW} d; k={KS}")
    sample, S = E3.sample_products()
    tasks = [(i, sample.loc[i, "id"], sample.loc[i, "cat_id"], S[i]) for i in range(len(sample))]
    sims, tgts = [], []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for n, (a, b) in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            sims += a
            tgts += b
            if n % 50 == 0:
                print(f"  {n}/{len(tasks)} productos ({time.time() - t0:.0f} s)")
    sim, tgt = pd.DataFrame(sims), pd.DataFrame(tgts)
    sim.to_csv(C.RESULTS / "exp7_simulacion_por_producto.csv", index=False)
    tgt.to_csv(C.RESULTS / "exp7_objetivos_por_revision.csv", index=False)
    print(f"Revisiones sin pronostico del motor (skipped): {(tgt['status'] == 'skipped').sum()} de {len(tgt)}")

    # Resumen agregado por L, metodo y k
    agg = sim.groupby(["L_meses", "metodo", "k"], as_index=False)[["demanda", "atendido", "perdido", "inv_unid_dia", "pedidos"]].sum()
    days = (SIM_END - ORIGINS[0]).days
    agg["fill"] = agg["atendido"] / agg["demanda"]
    agg["perdidas_pct"] = 100 * agg["perdido"] / agg["demanda"]
    agg["dias_inv"] = (agg["inv_unid_dia"] / days) / (agg["demanda"] / days)
    sim["fill_prod"] = np.where(sim["demanda"] > 0, sim["atendido"] / sim["demanda"], np.nan)
    med = sim.groupby(["L_meses", "metodo", "k"], as_index=False)["fill_prod"].median().rename(columns={"fill_prod": "fill_mediana_producto"})
    agg = agg.merge(med, on=["L_meses", "metodo", "k"])
    agg.round(4).to_csv(C.RESULTS / "exp7_resumen.csv", index=False)
    print("\nResumen con k = 1 (regla del sistema sin stock de seguridad):")
    print(agg[agg["k"] == 1.0][["L_meses", "metodo", "fill", "perdidas_pct", "dias_inv", "fill_mediana_producto"]].round(3).to_string(index=False))

    # Frontera: dias de inventario para 90 % y 95 % de servicio
    fr_rows = []
    for (L, m), g in agg.groupby(["L_meses", "metodo"]):
        fr_rows.append({"L_meses": L, "metodo": m, "dias_inv_fill90": round(frontier(g, 0.90), 2),
                        "dias_inv_fill95": round(frontier(g, 0.95), 2)})
    fr = pd.DataFrame(fr_rows)
    fr.to_csv(C.RESULTS / "exp7_frontera.csv", index=False)
    print("\nDias de inventario necesarios para una tasa de servicio de 90 % y 95 %:")
    print(fr.to_string(index=False))

    # Pruebas pareadas por producto con k = 1
    s1 = sim[sim["k"] == 1.0].copy()
    s1["dias_inv_prod"] = np.where(s1["demanda"] > 0, s1["inv_unid_dia"] / s1["demanda"], np.nan)
    tests = []
    for L in LENGTHS:
        g = s1[s1["L_meses"] == L]
        for ref in ("Naive", "SNaive"):
            for metric, better in (("fill_prod", "mayor"), ("perdido", "menor"), ("dias_inv_prod", "menor")):
                a = g[g["metodo"] == "Motor"].set_index("producto")[metric]
                b = g[g["metodo"] == ref].set_index("producto")[metric]
                idx = a.dropna().index.intersection(b.dropna().index)
                d = (a[idx] - b[idx]).to_numpy()
                nz = d[d != 0]
                p = float(wilcoxon(nz).pvalue) if nz.size > 0 else 1.0
                wins = (d > 0).mean() if better == "mayor" else (d < 0).mean()
                tests.append({"L_meses": L, "comparacion": f"Motor vs {ref}", "metrica": metric, "mejor_si": better,
                              "n": len(idx), "motor_mejor_pct": round(100 * wins, 1), "empate_pct": round(100 * (d == 0).mean(), 1),
                              "mediana_dif": round(float(np.median(d)), 4), "wilcoxon_p": p})
    te = pd.DataFrame(tests)
    te["p_holm"] = holm(te["wilcoxon_p"].to_numpy())
    te.to_csv(C.RESULTS / "exp7_pruebas_pareadas.csv", index=False)
    print("\nPruebas pareadas (k = 1), p con correccion de Holm sobre las", len(te), "pruebas:")
    print(te.to_string(index=False))
    print(f"\nTiempo total: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
