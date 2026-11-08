"""JISEBI copy at 600 dpi. English-labelled result figures for the papers, read-only from experimentos/results/*.csv.

No number is computed here that is not already in experimentos/results or experimentos/fidelidad/results;
the script only re-plots them with English labels. Run from any directory:

    python make_result_figures_en.py

Outputs (entregables/papers/figures/):
    fig_fidelity_gap_en.png      MASE h=1-15 (mean) of FTGM variants and baselines vs. Ye et al. Table 1
    fig_exp1_mase_horizon_en.png MASE by horizon (mean and median), with Ye et al. Table 1 FTGM markers
    fig_exp1_rmsse_box_en.png    RMSSE h=1-15 per series (boxplot)
    fig_exp2_grey_grades_en.png  Grey accuracy grades (C, p)
    fig_exp3_winner_en.png       Engine winner by history length
    fig_exp3_wape_en.png         Per-period WAPE of the engine and references
    fig_exp3b_censoring_en.png   Injected-censoring ablation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve()
OUT = HERE.parent / "paper_figures"
ROOT = HERE.parent
RES = ROOT / "results"
FID = ROOT / "fidelidad" / "results"

PALETTE = ["#2F5D8A", "#D08C3C", "#5B9A6B", "#A44A4A", "#7A6BA8", "#8C8C8C", "#3E8E9C", "#B5A642", "#C06C84"]
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 600, "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlesize": 10, "legend.frameon": False,
})

LABEL = {
    "FTGM_OLS_paper": "FTGM OLS (literal Alg. 1)", "FTGM_OLS_refit": "FTGM OLS (refit)",
    "FTGM_Ridge": "FTGM ridge (engine)", "FTGM_OLS_ancla": "FTGM OLS anchored (diagnostic)",
    "Motor_torneo": "Engine (model selection)", "SNaive": "Seasonal naive",
    "Naive": "Naive", "ETS": "ETS (AICc)", "SARIMA": "SARIMA (AICc)",
}
ORDER = list(LABEL)
# Ye et al. (2024), Table 1, FTGM row (mean/median) at h = 1, 3, 6, 9, 12, 15 (as transcribed in RESULTADOS.md)
PAPER_H = [1, 3, 6, 9, 12, 15]
PAPER_MEAN = [1.09, 1.24, 1.30, 2.04, 2.02, 1.99]
PAPER_MED = [0.82, 1.00, 1.02, 1.23, 1.28, 1.20]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


def fig_fidelity_gap():
    fid = pd.read_csv(FID / "fid_tabla_consolidada.csv").set_index("modelo")
    per = pd.read_csv(RES / "exp1_metricas_por_serie.csv", dtype={"h": str})
    s15 = per[per["h"] == "1-15"].groupby("modelo")["MASE"].agg(["mean", "median"])
    rows = [
        ("Ye et al. (2024), Table 1", 1.65, 1.28, "paper"),
        ("FTGM literal Alg. 1 (engine)", s15.loc["FTGM_OLS_paper", "mean"], s15.loc["FTGM_OLS_paper", "median"], "ftgm"),
        ("FTGM authors' MATLAB scheme", fid.loc["init_matlab_literal", "MASE_media"], fid.loc["init_matlab_literal", "MASE_mediana"], "ftgm"),
        ("FTGM OLS refit", s15.loc["FTGM_OLS_refit", "mean"], s15.loc["FTGM_OLS_refit", "median"], "ftgm"),
        ("FTGM ridge (engine)", s15.loc["FTGM_Ridge", "mean"], s15.loc["FTGM_Ridge", "median"], "ftgm"),
        ("FTGM restart at origin (not in paper)", fid.loc["origen_x_diff_refit", "MASE_media"], fid.loc["origen_x_diff_refit", "MASE_mediana"], "ftgm"),
        ("Engine (model selection)", s15.loc["Motor_torneo", "mean"], s15.loc["Motor_torneo", "median"], "engine"),
        ("Seasonal naive", s15.loc["SNaive", "mean"], s15.loc["SNaive", "median"], "base"),
        ("ETS (AICc)", s15.loc["ETS", "mean"], s15.loc["ETS", "median"], "base"),
        ("SARIMA (AICc)", s15.loc["SARIMA", "mean"], s15.loc["SARIMA", "median"], "base"),
        ("Leakage diagnostic: estimated with test window", fid.loc["oraculo_matlab", "MASE_media"], fid.loc["oraculo_matlab", "MASE_mediana"], "leak"),
    ]
    col = {"paper": "#222222", "ftgm": PALETTE[0], "engine": PALETTE[2], "base": PALETTE[5], "leak": PALETTE[3]}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, mean, med, kind) in zip(y, rows):
        ax.barh(yi, mean, 0.6, color=col[kind], alpha=0.85 if kind != "leak" else 0.45,
                hatch="//" if kind == "leak" else None, edgecolor="white")
        ax.plot(med, yi, marker="D", color="black", ms=3.5, zorder=5)
        ax.text(mean + 0.06, yi, f"{mean:.2f} / {med:.2f}", va="center", fontsize=7.5)
    ax.axvline(1.65, color="#222", lw=0.8, ls="--")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("MASE, h = 1-15 (bar: mean; diamond: median; 70 M5 series)")
    ax.set_title("RQ1: published FTGM accuracy vs. replication variants and baselines")
    ax.set_xlim(0, 5.6)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_fidelity_gap_en.png")


def fig_exp1():
    per = pd.read_csv(RES / "exp1_metricas_por_serie.csv", dtype={"h": str})
    s15 = per[per["h"] == "1-15"]
    data = [s15.loc[s15["modelo"] == m, "RMSSE"].values for m in ORDER]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops={"color": "#222", "lw": 1.2}, flierprops={"markersize": 3, "alpha": 0.5})
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, d in enumerate(data, 1):
        ax.text(i, ax.get_ylim()[1] * 0.97, f"{np.median(d):.2f}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(range(1, len(ORDER) + 1), [LABEL[m] for m in ORDER], rotation=25, ha="right")
    ax.set_ylabel("RMSSE, h = 1-15 (per series)")
    ax.set_title("Exp. 1: RMSSE per model on the 70 M5 department-store series")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_exp1_rmsse_box_en.png")

    ph = per[per["h"] != "1-15"].copy()
    ph["h"] = ph["h"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, stat, name, pv in ((axes[0], "mean", "Mean", PAPER_MEAN), (axes[1], "median", "Median", PAPER_MED)):
        agg = ph.groupby(["modelo", "h"])["MASE"].agg(stat).reset_index()
        for m, c in zip(ORDER, PALETTE):
            a = agg[agg["modelo"] == m]
            lw = 2.0 if m == "FTGM_OLS_paper" else 1.1
            ax.plot(a["h"], a["MASE"], marker="o", ms=2.5, lw=lw, color=c, label=LABEL[m])
        ax.plot(PAPER_H, pv, ls="none", marker="*", ms=9, color="black", label="FTGM, Ye et al. (2024) Table 1")
        ax.set_xlabel("Horizon h (months)")
        ax.set_ylabel(f"{name} MASE (70 series)")
        ax.set_title(f"{name} MASE by horizon")
        ax.set_xticks(range(1, 16))
    hnd, lbl = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(hnd, lbl, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_exp1_mase_horizon_en.png")


def fig_exp2():
    g = pd.read_csv(RES / "exp2_grados.csv").set_index("grado")
    d = pd.read_csv(RES / "exp2_diagnostico_por_serie.csv")
    order = ["I-Excelente", "II-Bueno", "III-Calificado", "IV-No calificado"]
    labels = ["I Excellent", "II Good", "III Qualified", "IV Not\nqualified"]
    models = ["FTGM_OLS", "FTGM_Ridge", "SNaive_ref(k>=13)"]
    mlabels = ["FTGM (OLS, paper)", "FTGM (ridge, engine)", "Seasonal naive (reference)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(order))
    w = 0.27
    for j, (m, ml) in enumerate(zip(models, mlabels)):
        counts = [int(g.loc[k, m]) if k in g.index else 0 for k in order]
        bars = axes[0].bar(x + (j - 1) * w, counts, w, label=ml, color=PALETTE[j])
        for b, c in zip(bars, counts):
            if c:
                axes[0].text(b.get_x() + b.get_width() / 2, c + 0.5, str(c), ha="center", fontsize=7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Number of series (of 70)")
    axes[0].set_title("In-sample grey accuracy grade (C and p)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", visible=False)
    for j, (m, ml) in enumerate(zip(models, mlabels)):
        sub = d[d["modelo"] == m]
        axes[1].hist(sub["C"], bins=np.linspace(0, 2.0, 41), alpha=0.6, label=ml, color=PALETTE[j])
    for thr in (0.35, 0.5, 0.65):
        axes[1].axvline(thr, color="#444", lw=0.8, ls="--")
        axes[1].text(thr + 0.01, axes[1].get_ylim()[1] * 0.92, f"{thr:.2f}", fontsize=7)
    axes[1].set_xlabel("Posterior variance ratio C = S2/S1")
    axes[1].set_ylabel("Number of series")
    axes[1].set_title("Distribution of C over the 70 series")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_exp2_grey_grades_en.png")


FAM = {"Familia FTGM": "FTGM family", "Croston-SBA": "Croston-SBA", "Promedio movil": "Moving average",
       "Suavizado amortiguado": "Damped trend", "Suavizado estacional": "Seasonal damped",
       "Naive estacional": "Seasonal naive", "Omitido": "Skipped"}


def fig_exp3():
    share = pd.read_csv(RES / "exp3_ganador_por_largo.csv")
    fams = [f for f in FAM if f in share["familia"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, H in zip(axes, sorted(share["H_dias"].unique())):
        p = share[share["H_dias"] == H].pivot_table(index="L_meses", columns="familia", values="pct", fill_value=0)
        bottom = np.zeros(len(p))
        x = np.arange(len(p))
        for f, c in zip(fams, PALETTE):
            v = p[f].values if f in p else np.zeros(len(p))
            ax.bar(x, v, 0.65, bottom=bottom, color=c, label=FAM[f])
            for xi, (vi, bi) in enumerate(zip(v, bottom)):
                if vi >= 6:
                    ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center", fontsize=7, color="white")
            bottom += v
        ax.set_xticks(x, [f"{L}" for L in p.index])
        ax.set_xlabel("Available history (months)")
        ax.set_title(f"Horizon {H} days")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of products (n = 300)")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Exp. 3: winning model of the engine by history length", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_exp3_winner_en.png")

    err = pd.read_csv(RES / "exp3_error_por_largo.csv")
    lab = {"Motor": "Engine (production)", "SNaive": "Seasonal naive", "Naive": "Naive", "FTGM_solo": "FTGM alone"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, H in zip(axes, sorted(err["H_dias"].unique())):
        e = err[err["H_dias"] == H]
        for (m, l), c in zip(lab.items(), PALETTE):
            a = e[e["metodo"] == m].sort_values("L_meses")
            ax.plot(a["L_meses"], a["WAPE_periodos_pct"], marker="o", lw=1.6 if m == "Motor" else 1.1,
                    color=c, label=l)
        ax.set_xlabel("Available history (months)")
        ax.set_title(f"Horizon {H} days")
        ax.set_xticks(sorted(e["L_meses"].unique()))
    axes[0].set_ylabel("Per-period WAPE (%)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Exp. 3: error of the production pipeline vs. references", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_exp3_wape_en.png")

    cs = pd.read_csv(RES / "exp3_censura_resumen.csv")
    conds = ["A_original", "B_censurada_sin_marcar", "C_censurada_marcada"]
    clab = ["A: original (uncensored)", "B: censored, not flagged", "C: censored and flagged"]
    groups = cs[["L_meses", "H_dias"]].drop_duplicates().sort_values(["L_meses", "H_dias"]).values
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(groups))
    w = 0.26
    for j, (c, cl) in enumerate(zip(conds, clab)):
        vals = [float(cs[(cs["L_meses"] == L) & (cs["H_dias"] == H) & (cs["condicion"] == c)]["WAPE_periodos_pct"].iloc[0])
                for L, H in groups]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=[PALETTE[5], PALETTE[3], PALETTE[0]][j], label=cl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [f"{L} months / {H} days" for L, H in groups])
    ax.set_ylabel("Per-period WAPE (%)")
    ax.set_title("Exp. 3b: stockout censoring injected into M5 data (100 products)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(fontsize=7, loc="upper center", ncol=3)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_exp3b_censoring_en.png")


def fig_exp7():
    """Exp. 7 (paper: Exp. 4, decision simulation): service vs inventory frontier for the safety factor k."""
    agg = pd.read_csv(RES / "exp7_resumen.csv")
    ref = pd.read_csv(RES / "exp7b_referencias_extra_por_producto.csv")
    ref = ref[ref["L_meses"] == 6].groupby(["metodo", "k"], as_index=False)[["demanda", "atendido", "inv_unid_dia"]].sum()
    ref["fill"] = ref["atendido"] / ref["demanda"]
    ref["dias_inv"] = ref["inv_unid_dia"] / ref["demanda"]
    lab = {"Motor": "Engine (production)", "SNaive": "Seasonal naive", "Naive": "Naive", "Oraculo": "Perfect forecast"}
    col = {"Motor": PALETTE[0], "SNaive": PALETTE[1], "Naive": PALETTE[2], "Oraculo": "#222222"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, L in zip(axes, (6, 12, 24)):
        for m, l in lab.items():
            a = agg[(agg["L_meses"] == L) & (agg["metodo"] == m)].sort_values("k")
            ax.plot(a["dias_inv"], 100 * a["fill"], marker="o", ms=3, lw=1.6 if m == "Motor" else 1.1,
                    ls="--" if m == "Oraculo" else "-", color=col[m], label=l)
            k1 = a[a["k"] == 1.0]
            ax.plot(k1["dias_inv"], 100 * k1["fill"], marker="o", ms=7, mfc="white", mec=col[m], mew=1.4, ls="none")
        ma = ref[ref["metodo"] == "MA4s"].sort_values("k")
        ax.plot(ma["dias_inv"], 100 * ma["fill"], marker="o", ms=3, lw=1.1, ls="-.", color=PALETTE[4],
                label="Average of last 4 weeks")
        k1 = ma[ma["k"] == 1.0]
        ax.plot(k1["dias_inv"], 100 * k1["fill"], marker="o", ms=7, mfc="white", mec=PALETTE[4], mew=1.4, ls="none")
        ax.axhline(90, color="#888", lw=0.7, ls=":")
        ax.set_title(f"{L} months of initial history")
        ax.set_xlabel("Average inventory (days of demand)")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("Fill rate (%)")
    axes[0].set_ylim(50, 100)
    axes[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("Exp. 4: service and inventory of the order-up-to rule with each forecast (300 products; "
                 "hollow marker: k = 1)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_exp4_decision_en.png")


if __name__ == "__main__":
    fig_fidelity_gap()
    fig_exp1()
    fig_exp2()
    fig_exp3()
    fig_exp7()
