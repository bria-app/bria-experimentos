"""Experimento 2 - Diagnosticos clasicos de modelos grises (pedido del asesor).

Para cada una de las 70 series departamento-tienda (63 meses, misma construccion que el
Experimento 1) se ajusta el FTGM dentro de muestra con toda la historia:
  * FTGM_OLS  : Algoritmo 1 (validacion 6, N in 1..5, lambda = 0) y re-estimacion con 63 obs.
  * FTGM_Ridge: Algoritmo 1 del motor con rejilla ridge (0, 0.5, 3, 10).
  * Referencia: ajuste "naive estacional" dentro de muestra (x_hat_k = x_{k-12}, k >= 13).

Indicadores (ver common.grey_diagnostics): C = S2/S1, p = P(|e_k - e_medio| < 0.6745 S1),
error relativo medio (%), grado relacional de Deng (rho = 0.5). Grados de precision segun
Liu et al. (2025, doi:10.1186/s12879-025-11483-4, Tabla 2): I Excelente C<0.35 y P>=0.95;
II Bueno C<0.50 y P>=0.80; III Calificado C<0.65 y P>=0.70; IV No calificado.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time

import numpy as np
import pandas as pd

import common as C
from exp1_replicacion_ye import build_series

T = 12


def fit_ftgm(y: np.ndarray, ridges: tuple[float, ...]) -> tuple[np.ndarray, int, float, str]:
    from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order

    sel = select_order(y, FTGMConfig(period=T), validation_size=6, max_order=5, ridges=ridges)
    ranking = [sel.best_order] + [k for _, k in sorted((v, k) for k, v in sel.scores.items() if np.isfinite(v))
                                  if k != sel.best_order]
    for k in ranking:
        try:
            m = FTGM(order=k, config=FTGMConfig(period=T, ridge=sel.best_ridge)).fit(y)
            note = "" if k == sel.best_order else f"orden {sel.best_order} diverge"
            return m.fitted_, k, sel.best_ridge, note
        except FTGMError:
            continue
    raise RuntimeError("ningun orden estable")


def main() -> None:
    sys.stdout = C.Tee(C.LOGS / "exp2_log.txt")  # type: ignore[assignment]
    t0 = time.time()
    print("EXPERIMENTO 2 - Diagnosticos de modelo gris (ajuste dentro de muestra)")
    print(C.environment())
    names, Y, _ = build_series()
    rows = []
    for i, (name, y) in enumerate(zip(names, Y)):
        for label, ridges in (("FTGM_OLS", (0.0,)), ("FTGM_Ridge", (0.0, 0.5, 3.0, 10.0))):
            try:
                fitted, k, lam, note = fit_ftgm(y, ridges)
            except RuntimeError as exc:
                rows.append({"serie_id": i + 1, "serie": name, "modelo": label, "nota": str(exc)})
                continue
            d = C.grey_diagnostics(y, fitted)
            rows.append({"serie_id": i + 1, "serie": name, "modelo": label, "orden": k, "ridge": lam,
                         **d, "grado": C.grey_grade(d["C"], d["p"]), "nota": note})
        fitted = y[:-T]
        d = C.grey_diagnostics(y[T:], fitted)
        rows.append({"serie_id": i + 1, "serie": name, "modelo": "SNaive_ref(k>=13)", "orden": None, "ridge": None,
                     **d, "grado": C.grey_grade(d["C"], d["p"]), "nota": ""})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "exp2_diagnostico_por_serie.csv", index=False)

    summ = df.groupby("modelo")[["C", "p", "RRM_pct", "GRD"]].agg(["mean", "median", "min", "max"]).round(4)
    print("\nResumen de indicadores (70 series):")
    print(summ.to_string())
    summ.to_csv(C.RESULTS / "exp2_resumen_indicadores.csv")
    grades = df.pivot_table(index="grado", columns="modelo", values="serie", aggfunc="count", fill_value=0)
    print("\nNumero de series por grado de precision:")
    print(grades.to_string())
    grades.to_csv(C.RESULTS / "exp2_grados.csv")
    for m in df["modelo"].unique():
        sub = df[df["modelo"] == m]
        print(f"{m}: GRD >= 0.6 en {(sub['GRD'] >= 0.6).sum()} de {len(sub)} series; "
              f"divergencias del orden elegido: {(sub['nota'].fillna('') != '').sum()}")

    # ------------------------------------------------------------------- figuras
    import matplotlib.pyplot as plt

    C.plot_style()
    order = ["I-Excelente", "II-Bueno", "III-Calificado", "IV-No calificado"]
    labels = ["I Excelente", "II Bueno", "III Calificado", "IV No\ncalificado"]
    models = ["FTGM_OLS", "FTGM_Ridge", "SNaive_ref(k>=13)"]
    mlabels = ["FTGM (MCO, paper)", "FTGM (ridge, motor)", "Naive estacional (referencia)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(order))
    w = 0.27
    for j, (m, ml) in enumerate(zip(models, mlabels)):
        counts = [int(grades.loc[g, m]) if g in grades.index else 0 for g in order]
        bars = axes[0].bar(x + (j - 1) * w, counts, w, label=ml, color=C.PALETTE[j])
        for b, c in zip(bars, counts):
            if c:
                axes[0].text(b.get_x() + b.get_width() / 2, c + 0.5, str(c), ha="center", fontsize=7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Número de series (de 70)")
    axes[0].set_title("Grado de precisión del ajuste (C y p)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", visible=False)
    for j, (m, ml) in enumerate(zip(models, mlabels)):
        sub = df[df["modelo"] == m]
        axes[1].hist(sub["C"], bins=np.linspace(0, 2.0, 41), alpha=0.6, label=ml, color=C.PALETTE[j])
    for thr, txt in ((0.35, "0,35"), (0.5, "0,50"), (0.65, "0,65")):
        axes[1].axvline(thr, color="#444", lw=0.8, ls="--")
        axes[1].text(thr + 0.01, axes[1].get_ylim()[1] * 0.92, txt, fontsize=7)
    axes[1].set_xlabel("Razón de varianza posterior C = S2/S1")
    axes[1].set_ylabel("Número de series")
    axes[1].set_title("Distribución de C en las 70 series")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_exp2_grados_gris.png")
    print(f"\nTiempo total: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
