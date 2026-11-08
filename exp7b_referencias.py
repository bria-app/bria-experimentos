"""Exp. 7b - Referencias suavizadas para la simulacion de decisiones (paper JISEBI: Tabla 10).

Mismos productos, revisiones, regla y metricas que exp7_simulacion_decision.py, con tres
pronosticos que un comerciante calcula a mano: ventas del ultimo mes, promedio de las ultimas
4 semanas y promedio de los ultimos 3 meses (escalados a los 37 dias de plazo + revision).
No dependen de la historia disponible. Compara por producto la tasa de servicio del motor
(k = 1, results/exp7_simulacion_por_producto.csv) con cada referencia (Wilcoxon + Holm).

Salidas: results/exp7b_referencias_extra_por_producto.csv, exp7b_referencias_extra_resumen.csv,
exp7b_pruebas_motor_vs_referencias.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C
import exp3_series_cortas as E3
import exp7_simulacion_decision as X

NAMES = {"NaiveMensual": "NaiveMensual", "MA4s": "Promedio4s", "MA3m": "Promedio3m"}


def main() -> None:
    sample, S = E3.sample_products()
    w = X.LEAD + X.REVIEW
    rows = []
    for i in range(len(sample)):
        full = S[i]
        daily = full[X.di(X.ORIGINS[0]) : X.di(X.SIM_END)].astype(float)
        tg = {"NaiveMensual": [], "MA3m": [], "MA4s": []}
        for o in X.ORIGINS:
            pm, p3 = C.add_months(o, -1), C.add_months(o, -3)
            tg["NaiveMensual"].append(full[X.di(pm) : X.di(o)].sum() / (o - pm).days * w)
            tg["MA3m"].append(full[X.di(p3) : X.di(o)].sum() / (o - p3).days * w)
            tg["MA4s"].append(full[X.di(o) - 28 : X.di(o)].sum() / 28 * w)
        for m, t in tg.items():
            for k in X.KS:
                r = X.simulate(daily, t, k)
                for L in X.LENGTHS:
                    rows.append({"producto": i, "L_meses": L, "metodo": m, "k": k, **r})
    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / "exp7b_referencias_extra_por_producto.csv", index=False)
    d1 = d[d.L_meses == 6]
    agg = d1.groupby(["metodo", "k"], as_index=False)[["demanda", "atendido", "inv_unid_dia"]].sum()
    agg["fill"] = agg.atendido / agg.demanda
    agg["dias_inv"] = agg.inv_unid_dia / agg.demanda
    res = []
    for m, g in agg.groupby("metodo"):
        k1 = g[g.k == 1.0].iloc[0]
        res.append({"metodo": m, "fill_k1": round(100 * k1.fill, 1), "dias_k1": round(k1.dias_inv, 1),
                    "dias_fill90": round(X.frontier(g, 0.9), 1)})
    pd.DataFrame(res).to_csv(C.RESULTS / "exp7b_referencias_extra_resumen.csv", index=False)
    print(pd.DataFrame(res).to_string(index=False))
    e = pd.read_csv(C.RESULTS / "exp7_simulacion_por_producto.csv")
    t = []
    for L in X.LENGTHS:
        a = e[(e.L_meses == L) & (e.metodo == "Motor") & (e.k == 1.0)].set_index("producto")
        for m, label in NAMES.items():
            b = d1[(d1.metodo == m) & (d1.k == 1.0)].set_index("producto")
            fa, fb = a.atendido / a.demanda, b.atendido / b.demanda
            idx = fa.dropna().index.intersection(fb.dropna().index)
            diff = (fa[idx] - fb[idx]).to_numpy()
            nz = diff[diff != 0]
            t.append({"L": L, "ref": label, "motor_mejor_pct": round(100 * (diff > 0).mean(), 1),
                      "p": float(wilcoxon(nz).pvalue)})
    t = pd.DataFrame(t)
    t["p_holm"] = X.holm(t.p.to_numpy())
    t.to_csv(C.RESULTS / "exp7b_pruebas_motor_vs_referencias.csv", index=False)
    print(t.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
