"""Correccion de Holm para las pruebas pareadas del Exp. 3 (revision del paper JISEBI, 07/11/2026).

Lee results/exp3_pruebas_pareadas.csv (sin modificarlo), quita las filas duplicadas en las que el
naive estacional coincide con el naive (historias de 6 y 9 meses) y ajusta, como una sola familia,
los p de las dos metricas (error medio por periodo y error del total del horizonte).
Salida: results/exp3_pruebas_pareadas_holm.csv
"""
import numpy as np
import pandas as pd

import common as C
from exp7_simulacion_decision import holm

t = pd.read_csv(C.RESULTS / "exp3_pruebas_pareadas.csv")
long = pd.concat([
    t.assign(metrica="MAE_periodo", p=t["wilcoxon_p_MAE_periodo"], motor_gana=t["motor_gana_pct"]),
    t.assign(metrica="error_total", p=t["wilcoxon_p_error_total"], motor_gana=t["motor_gana_total_pct"]),
])[["H_dias", "L_meses", "comparacion", "metrica", "n", "motor_gana", "p"]]
long = long.drop_duplicates(subset=["H_dias", "L_meses", "metrica", "n", "motor_gana", "p"])
long["p_holm"] = holm(long["p"].to_numpy())
long["signif_holm_005"] = long["p_holm"] < 0.05
long.sort_values(["metrica", "comparacion", "H_dias", "L_meses"]).to_csv(C.RESULTS / "exp3_pruebas_pareadas_holm.csv", index=False)
print(f"{len(long)} pruebas en la familia")
print(long.sort_values(["metrica", "comparacion", "H_dias", "L_meses"]).round(5).to_string(index=False))

# Exp. 1: victorias frente a SNaive (16 pruebas) y Exp. 3b: censura (8 pruebas), cada una como familia.
e1 = pd.read_csv(C.RESULTS / "exp1_victorias_vs_snaive.csv")
e1["p_holm"] = holm(e1["wilcoxon_p"].to_numpy())
e1.to_csv(C.RESULTS / "exp1_victorias_vs_snaive_holm.csv", index=False)
cz = pd.read_csv(C.RESULTS / "exp3_censura_pruebas.csv")
adj = holm(np.r_[cz["wilcoxon_p"].to_numpy(), cz["wilcoxon_p_historia"].to_numpy()])
cz["p_holm"], cz["p_holm_historia"] = adj[: len(cz)], adj[len(cz):]
cz.to_csv(C.RESULTS / "exp3_censura_pruebas_holm.csv", index=False)
print("Exp. 1 y censura ajustados")
