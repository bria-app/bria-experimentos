"""Consolida results/fid_resumen.csv en una tabla 1-15 con brecha relativa frente a la Tabla 1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P = {"MASE_media": 1.65, "MASE_mediana": 1.28, "RMSSE_media": 1.38, "RMSSE_mediana": 1.04}
CLASE = {"H7_cota_oraculo": "fuga (diagnostico)", "H8_orden_con_prueba": "fuga parcial (diagnostico)",
         "H10_error_de_ajuste": "no es pronostico (diagnostico)", "H9_reinicio_origen": "pronostico honesto, fuera del paper"}

r = pd.read_csv(HERE / "results" / "fid_resumen.csv", dtype={"h": str})
t = r[r["h"] == "1-15"].copy()
for k, v in P.items():
    t[f"{k}_brecha_pct"] = ((t[k] - v) / v * 100).round(1)
t["clase"] = t["grupo"].map(CLASE).fillna("pronostico honesto")
t["dentro_10pct_las_4"] = (t[[f"{k}_brecha_pct" for k in P]].abs() <= 10).all(axis=1)
cols = ["grupo", "modelo", "clase"] + list(P) + [f"{k}_brecha_pct" for k in P] + ["dentro_10pct_las_4"]
t = t[cols].sort_values("MASE_media")
for k in P:
    t[k] = t[k].round(2)
t.to_csv(HERE / "results" / "fid_tabla_consolidada.csv", index=False)
pd.set_option("display.width", 250)
txt = t.to_string(index=False)
print(txt)
(HERE / "logs" / "fid_consolidado_log.txt").write_text(txt + "\n", encoding="utf-8")

# Huella por serie: el paper (Sec. 5.1) dice que las RMSSE 1-15 mas altas del FTGM son las de las
# series 40, 39, 20 y 60 (3.22, 3.10, 3.04, 2.85). Serie 13 = Foods_2_CA_2 (salto de nivel ~4x en
# jun-2015, mes 53, dentro de todas las ventanas de prueba).
import glob  # noqa: E402

fs = sorted(glob.glob(str(HERE / "results" / "fid_metricas_por_serie_*.csv")))
e1 = pd.read_csv(HERE.parent / "results" / "exp1_metricas_por_serie.csv", dtype={"h": str})
m = pd.concat([pd.read_csv(f, dtype={"h": str}) for f in fs] +
              [e1[e1["modelo"].isin(["SNaive", "Naive", "ETS", "SARIMA", "Motor_torneo"])]])
pv = m[m["h"] == "1-15"].pivot_table(index="modelo", columns="serie_id", values="RMSSE")
hu = pv[[13, 40, 39, 20, 60]].round(2)
hu.columns = [f"serie_{c}" for c in hu.columns]
hu["max_70"] = pv.max(axis=1).round(2)
hu["serie_del_max"] = pv.idxmax(axis=1)
hu["media_70"] = pv.mean(axis=1).round(2)
hu["media_sin_13"] = pv.drop(columns=13).mean(axis=1).round(2)
hu = hu.sort_values("media_70")
hu.to_csv(HERE / "results" / "fid_huella_por_serie.csv")
print("\nHuella por serie (RMSSE 1-15; paper FTGM: s40 3.22, s39 3.10, s20 3.04, s60 2.85):")
print(hu.to_string())
with open(HERE / "logs" / "fid_consolidado_log.txt", "a", encoding="utf-8") as f:
    f.write("\nHuella por serie (RMSSE 1-15; paper FTGM: s40 3.22, s39 3.10, s20 3.04, s60 2.85):\n" + hu.to_string() + "\n")
