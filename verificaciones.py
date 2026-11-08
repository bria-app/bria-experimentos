"""Verificaciones auxiliares citadas en RESULTADOS.md (lee datos y results/*.csv; no re-ejecuta modelos).

1. Calendario M5: d_1 = 2011-01-29 (ventas ~0 los 25 de diciembre).
2. Exp. 3b: % de productos en que cambia el modelo ganador entre condiciones de censura y
   unidades recuperadas por la reparacion.
"""
from __future__ import annotations

import sys

import pandas as pd

import common as C


def main() -> None:
    sys.stdout = C.Tee(C.LOGS / "verificaciones_log.txt")  # type: ignore[assignment]
    _, sales = C.load_m5()
    print("1) Verificacion de calendario (venta total del M5 por dia):")
    for k, v in C.verify_calendar(sales).items():
        print(f"   {k}: {v:.0f}")

    c = pd.read_csv(C.RESULTS / "exp3_censura_por_producto.csv")
    print("\n2) Exp. 3b - cambio del modelo ganador entre condiciones (% de productos):")
    for (H, L), g in c.groupby(["H_dias", "L_meses"]):
        m = g.pivot(index="producto", columns="condicion", values="modelo_motor")
        ab = (m["A_original"] != m["B_censurada_sin_marcar"]).mean() * 100
        ac = (m["A_original"] != m["C_censurada_marcada"]).mean() * 100
        print(f"   H={H} L={L}: A->B {ab:.1f} %, A->C {ac:.1f} %")
    print("\n3) Exp. 3b - unidades recuperadas por la reparacion (serie limpiada C - serie limpiada B):")
    for L, g in c[c["H_dias"] == 30].groupby("L_meses"):
        p = g.pivot(index="producto", columns="condicion", values="hist_reparada")
        rec = float((p["C_censurada_marcada"] - p["B_censurada_sin_marcar"]).sum())
        lost = float(g.groupby("producto")["unidades_censuradas"].first().sum())
        print(f"   L={L}: recuperadas {rec:.1f} de {lost:.0f} censuradas ({100 * rec / lost:.1f} %)")


if __name__ == "__main__":
    main()
