"""Figuras de los experimentos 1 y 3 (300 dpi, etiquetas en espanol) a partir de results/*.csv.

La figura del Experimento 2 la genera exp2_diagnostico_gris.py.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

C.plot_style()
R = C.RESULTS
LABEL = {
    "FTGM_OLS_paper": "FTGM MCO (paper)", "FTGM_OLS_refit": "FTGM MCO (re-estimado)",
    "FTGM_Ridge": "FTGM ridge (motor)", "FTGM_OLS_ancla": "FTGM MCO anclado (diagnóstico)",
    "Motor_torneo": "Motor (torneo)", "SNaive": "Naive estacional",
    "Naive": "Naive", "ETS": "ETS (AICc)", "SARIMA": "SARIMA (AICc)",
}
ORDER = list(LABEL)


def fig_exp1() -> None:
    per = pd.read_csv(R / "exp1_metricas_por_serie.csv", dtype={"h": str})
    s15 = per[per["h"] == "1-15"]
    data = [s15.loc[s15["modelo"] == m, "RMSSE"].values for m in ORDER]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops={"color": "#222", "lw": 1.2}, flierprops={"markersize": 3, "alpha": 0.5})
    for patch, col in zip(bp["boxes"], C.PALETTE):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, d in enumerate(data, 1):
        ax.text(i, ax.get_ylim()[1] * 0.97, f"{np.median(d):.2f}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(range(1, len(ORDER) + 1), [LABEL[m] for m in ORDER], rotation=25, ha="right")
    ax.set_ylabel("RMSSE promedio h = 1–15 (por serie)")
    ax.set_title("Exp. 1: RMSSE por modelo en las 70 series departamento-tienda del M5")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_exp1_rmsse_boxplot.png")
    plt.close(fig)

    ph = per[per["h"] != "1-15"].copy()
    ph["h"] = ph["h"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, stat, name in ((axes[0], "mean", "Media"), (axes[1], "median", "Mediana")):
        agg = ph.groupby(["modelo", "h"])["MASE"].agg(stat).reset_index()
        for m, col in zip(ORDER, C.PALETTE):
            a = agg[agg["modelo"] == m]
            lw = 2.0 if m.startswith("FTGM_OLS_paper") else 1.1
            ax.plot(a["h"], a["MASE"], marker="o", ms=2.5, lw=lw, color=col, label=LABEL[m])
        ax.set_xlabel("Horizonte h (meses)")
        ax.set_ylabel(f"{name} de MASE (70 series)")
        ax.set_title(f"{name} de MASE por horizonte")
        ax.set_xticks(range(1, 16))
    axes[1].legend(fontsize=7, ncol=1, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_exp1_mase_horizonte.png")
    plt.close(fig)


def fig_exp3() -> None:
    share = pd.read_csv(R / "exp3_ganador_por_largo.csv")
    fams = ["Familia FTGM", "Croston-SBA", "Promedio movil", "Suavizado amortiguado", "Suavizado estacional",
            "Naive estacional", "Omitido"]
    fams = [f for f in fams if f in share["familia"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, H in zip(axes, sorted(share["H_dias"].unique())):
        p = share[share["H_dias"] == H].pivot_table(index="L_meses", columns="familia", values="pct", fill_value=0)
        bottom = np.zeros(len(p))
        x = np.arange(len(p))
        for f, col in zip(fams, C.PALETTE):
            v = p[f].values if f in p else np.zeros(len(p))
            ax.bar(x, v, 0.65, bottom=bottom, color=col, label=f)
            for xi, (vi, bi) in enumerate(zip(v, bottom)):
                if vi >= 6:
                    ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=7, color="white")
            bottom += v
        ax.set_xticks(x, [f"{L}" for L in p.index])
        ax.set_xlabel("Historia disponible (meses)")
        ax.set_title(f"Horizonte {H} días")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% de productos (n = 300)")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Exp. 3: modelo ganador del torneo del motor según largo de historia", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_exp3_ganador_por_largo.png")
    plt.close(fig)

    err = pd.read_csv(R / "exp3_error_por_largo.csv")
    lab = {"Motor": "Motor (producción)", "SNaive": "Naive estacional", "Naive": "Naive", "FTGM_solo": "FTGM solo"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, H in zip(axes, sorted(err["H_dias"].unique())):
        e = err[err["H_dias"] == H]
        for (m, l), col in zip(lab.items(), C.PALETTE):
            a = e[e["metodo"] == m].sort_values("L_meses")
            ax.plot(a["L_meses"], a["WAPE_periodos_pct"], marker="o", lw=1.6 if m == "Motor" else 1.1,
                    color=col, label=l)
        ax.set_xlabel("Historia disponible (meses)")
        ax.set_title(f"Horizonte {H} días")
        ax.set_xticks(sorted(e["L_meses"].unique()))
    axes[0].set_ylabel("WAPE por periodo (%)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Exp. 3: error del pipeline de producción frente a referencias", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_exp3_wape_por_largo.png")
    plt.close(fig)

    cs = pd.read_csv(R / "exp3_censura_resumen.csv")
    conds = ["A_original", "B_censurada_sin_marcar", "C_censurada_marcada"]
    clab = ["A: original (sin censura)", "B: censurada, sin marcar", "C: censurada y marcada"]
    groups = cs[["L_meses", "H_dias"]].drop_duplicates().sort_values(["L_meses", "H_dias"]).values
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(groups))
    w = 0.26
    for j, (c, cl) in enumerate(zip(conds, clab)):
        vals = [float(cs[(cs["L_meses"] == L) & (cs["H_dias"] == H) & (cs["condicion"] == c)]["WAPE_periodos_pct"].iloc[0])
                for L, H in groups]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=[C.PALETTE[5], C.PALETTE[3], C.PALETTE[0]][j], label=cl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [f"{L} meses / {H} días" for L, H in groups])
    ax.set_ylabel("WAPE por periodo (%)")
    ax.set_title("Exp. 3b: censura por quiebre inyectada sobre datos M5 (100 productos)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(fontsize=7, loc="upper center", ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_exp3_censura_ablacion.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["1", "3"]
    if "1" in which:
        fig_exp1()
    if "3" in which:
        fig_exp3()
    print("figuras generadas:", sorted(p.name for p in C.FIGURES.glob("*.png")))
