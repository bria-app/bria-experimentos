"""Corre las variantes del FTGM (H1-H6) con el protocolo del Exp. 1 y resume contra la Tabla 1.

Uso:  python fid_run.py [grupo ...]      (sin argumentos corre todos los grupos)
Salidas: results/fid_crudos_<grupo>.csv, results/fid_resumen.csv (acumulativo), logs/fid_<grupo>_log.txt
Reproducible: sin aleatoriedad (semilla fija 42 declarada por consistencia), datos de
../results/exp1_series_mensuales.csv (generado por exp1_replicacion_ye.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import ftgm_variantes as V  # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(C.SEED)
H = 15
ORIGINS = list(range(43, 49))
RES = HERE / "results"
LOGS = HERE / "logs"
RES.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

PAPER_FTGM = {"MASE": {1: (1.09, 0.82), 3: (1.24, 1.00), 6: (1.30, 1.02), 9: (2.04, 1.23), 12: (2.02, 1.28),
                       15: (1.99, 1.20), "1-15": (1.65, 1.28)},
              "RMSSE": {1: (0.92, 0.73), 3: (1.05, 0.86), 6: (1.08, 0.82), 9: (1.65, 1.11), 12: (1.69, 0.99),
                        15: (1.67, 1.06), "1-15": (1.38, 1.04)}}


def B(**kw):
    return {**V.BASE, **kw}


LIT = dict(val=6, nmax=5, crit="rmse", final="train")   # Algoritmo 1 literal
REF = dict(val=6, nmax=5, crit="rmse", final="refit")   # re-estimacion con todo el in-sample

# grupo -> {variante: (cfg, sel)}
GROUPS: dict[str, dict[str, tuple[dict, dict]]] = {
    "H0_base": {
        "base_literal": (B(), LIT),
        "base_refit": (B(), REF),
    },
    "H1_restauracion": {
        "diff_literal": (B(restore="diff"), LIT),
        "diff_refit": (B(restore="diff"), REF),
    },
    "H2_inicial_tiempo": {
        "init_x1_refit": (B(init="x1"), REF),
        "init_matlab_refit": (B(est="matlab", init="matlab"), REF),
        "init_matlab_literal": (B(est="matlab", init="matlab"), LIT),
        "t0_0_refit": (B(t0=0.0), REF),
        "t0_0.5_refit": (B(t0=0.5), REF),
        "h_1/12_T1_refit": (B(h=1 / 12, T=1.0), REF),        # tiempo en anos, w = 2*pi
        "h_1/12_T12_refit": (B(h=1 / 12, T=12.0), REF),      # T mal escalado (w = 2*pi/12 por ano)
        "T_2pi_refit": (B(T=2 * np.pi), REF),                # w = 1 (error tipico de w)
    },
    "H3_seleccion": {
        "val12_refit": (B(), dict(REF, val=12)),
        "val15_refit": (B(), dict(REF, val=15)),
        "val3_refit": (B(), dict(REF, val=3)),
        "nmax3_refit": (B(), dict(REF, nmax=3)),
        "nmax6_refit": (B(), dict(REF, nmax=6)),
        "mape_refit": (B(), dict(REF, crit="mape")),
        "mae_refit": (B(), dict(REF, crit="mae")),
        "fitrmse_refit": (B(), dict(REF, crit="fit_rmse")),
        "N1_fijo_refit": (B(), dict(REF, fixed_N=1)),
        "N2_fijo_refit": (B(), dict(REF, fixed_N=2)),
    },
    "H6_combinadas": {
        "matlab_literal_val12": (B(est="matlab", init="matlab"), dict(LIT, val=12)),
        "matlab_refit_N1": (B(est="matlab", init="matlab"), dict(REF, fixed_N=1)),
        "matlab_refit_nmax3": (B(est="matlab", init="matlab"), dict(REF, nmax=3)),
        "matlab_refit_diff": (B(est="matlab", init="matlab", restore="diff"), REF),
        "matlab_literal_diff": (B(est="matlab", init="matlab", restore="diff"), LIT),
        "matlab_refit_ridge3": (B(est="matlab", init="matlab", ridge=3.0), REF),
    },
    "H7_cota_oraculo": {
        # DIAGNOSTICO con fuga deliberada: se estima con in-sample + los 15 meses de prueba y se
        # evalua el AJUSTE sobre la ventana de prueba. Es una cota inferior del error alcanzable
        # por la clase de modelo, no un pronostico.
        "oraculo_base": (B(), dict(REF, oracle=True)),
        "oraculo_matlab": (B(est="matlab", init="matlab"), dict(REF, oracle=True)),
        "oraculo_base_N5": (B(), dict(REF, oracle=True, fixed_N=5)),
    },
    "H8_orden_con_prueba": {
        # DIAGNOSTICO con fuga parcial: el orden N se elige con el RMSE de la ventana de PRUEBA
        # (los parametros si se estiman sin ver la prueba).
        "Ntest_base_literal": (B(), dict(LIT, test_select=True)),
        "Ntest_base_refit": (B(), dict(REF, test_select=True)),
        "Ntest_matlab_literal": (B(est="matlab", init="matlab"), dict(LIT, test_select=True)),
        "Ntest_matlab_refit": (B(est="matlab", init="matlab"), dict(REF, test_select=True)),
    },
    "H9_reinicio_origen": {
        # DIAGNOSTICO (no descrito en el paper): la EDO de 2 estados se reinicia en el origen con el
        # ultimo dato observado (x) y la acumulada observada (y).
        "origen_x_base_refit": (B(init="origin_x"), REF),
        "origen_x_base_literal": (B(init="origin_x"), LIT),
        "origen_x_matlab_refit": (B(est="matlab", init="origin_x"), REF),
        "origen_y_base_refit": (B(init="origin_y"), REF),
        "origen_x_base_refit_N1": (B(init="origin_x"), dict(REF, fixed_N=1)),
        "origen_x_diff_refit": (B(init="origin_x", restore="diff"), REF),
    },
    "H10_error_de_ajuste": {
        # DIAGNOSTICO de una posible confusion de evaluacion: se "evaluan" como pronostico los
        # valores AJUSTADOS de los ultimos 15 meses del in-sample (frente a sus reales), escalados
        # igual. No es un pronostico; mide cuanto se pareceria la Tabla 1 a un error de ajuste.
        "ajuste_ult15_matlab_refit": (B(est="matlab", init="matlab"), dict(REF, insample_fit=True)),
        "ajuste_ult15_base_refit": (B(), dict(REF, insample_fit=True)),
    },
    "H2b_ode45_2estados": {
        # Sistema de 2 estados integrado directamente con las tolerancias por defecto de ode45
        # (RelTol 1e-3, AbsTol 1e-6) y con tolerancias estrictas (control).
        "matlab2s_tol1e-3_literal": (B(est="matlab", init="matlab", two_state=True, rtol=1e-3, atol=1e-6), LIT),
        "matlab2s_tol1e-3_refit": (B(est="matlab", init="matlab", two_state=True, rtol=1e-3, atol=1e-6), REF),
        "matlab2s_tol1e-9_literal": (B(est="matlab", init="matlab", two_state=True, rtol=1e-9, atol=1e-6), LIT),
        "paper2s_tol1e-3_refit": (B(two_state=True, rtol=1e-3, atol=1e-6), REF),
    },
    "H4_estimacion": {
        "nox1_refit": (B(est="nox1"), REF),
        "matlab_est_paperinit_refit": (B(est="matlab"), REF),
        "matlab_eta_eval_refit": (B(est="matlab", init="matlab", eta_mode="eval"), REF),
    },
}


def run_one(args):
    warnings.filterwarnings("ignore")
    i, name, y, variants = args
    rows, infos = [], []
    for o in ORIGINS:
        ins, act0 = y[:o], y[o:o + H]
        s_mae, s_mse = C.paper_scales(ins)
        for vname, (cfg, sel) in variants.items():
            act = act0
            if sel.get("insample_fit"):
                fc, info = insample_fit(ins, cfg, sel)
                act = ins[-H:]
            elif sel.get("test_select"):
                fc, info = test_select(ins, act, cfg, sel)
            elif sel.get("oracle"):
                fc, info = oracle(y[:o + H], ins, cfg, sel)
            else:
                fc, info = V.select_and_forecast(ins, H, cfg, sel)
            for h in range(1, H + 1):
                rows.append({"serie_id": i + 1, "serie": name, "origen": o, "modelo": vname, "h": h,
                             "real": act[h - 1], "pronostico": float(fc[h - 1]), "escala_mae": s_mae,
                             "escala_mse": s_mse})
            infos.append({"serie_id": i + 1, "origen": o, "modelo": vname, **info})
    return rows, infos


def oracle(full, ins, cfg, sel):
    if sel.get("fixed_N"):
        N = sel["fixed_N"]
    else:
        _, info = V.select_and_forecast(ins, H, cfg, dict(sel, final="refit"))
        N = max(1, info["N"])
    for k in [N] + [j for j in range(1, 6) if j != N]:
        try:
            fit, _ = V.fit_forecast(full, k, 1, cfg)
            return np.clip(fit[-H:], 0, None), {"N": k, "fallo": "" if k == N else "otro N"}
        except V.Diverge:
            continue
    return V.select_and_forecast(ins, H, cfg, sel)


def test_select(ins, act, cfg, sel):
    best = None
    for N in range(1, sel.get("nmax", 5) + 1):
        fc, info = V.select_and_forecast(ins, H, cfg, dict(sel, fixed_N=N) if sel["final"] == "refit" else
                                         dict(sel, only_N=N))
        if info["fallo"]:
            continue
        r = float(np.sqrt(np.mean((act - fc) ** 2)))
        if best is None or r < best[0]:
            best = (r, fc, {"N": N, "fallo": ""})
    if best is None:
        return V.select_and_forecast(ins, H, cfg, sel)
    return best[1], best[2]


def insample_fit(ins, cfg, sel):
    _, info = V.select_and_forecast(ins, H, cfg, sel)
    for k in [max(1, info["N"])] + list(range(1, 6)):
        try:
            fit, _ = V.fit_forecast(ins, k, 1, cfg)
            return np.clip(fit[-H:], 0, None), {"N": k, "fallo": ""}
        except V.Diverge:
            continue
    return np.full(H, ins.mean()), {"N": 0, "fallo": "media"}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Misma agregacion que exp1_replicacion_ye.summarize (por horizonte: media de 6 origenes;
    1-15: metrica de 15 pasos por origen, media de origenes; luego media/mediana sobre 70 series)."""
    raw = raw.copy()
    raw["ae"] = (raw["real"] - raw["pronostico"]).abs()
    raw["se"] = (raw["real"] - raw["pronostico"]) ** 2
    raw["MASE"] = raw["ae"] / raw["escala_mae"]
    raw["RMSSE"] = np.sqrt(raw["se"] / raw["escala_mse"])
    per_h = raw.groupby(["modelo", "serie_id", "h"])[["MASE", "RMSSE"]].mean().reset_index()
    g = raw.groupby(["modelo", "serie_id", "origen"])
    o = pd.DataFrame({"MASE": g["ae"].mean() / g["escala_mae"].first(),
                      "RMSSE": np.sqrt(g["se"].mean() / g["escala_mse"].first())}).reset_index()
    ps = o.groupby(["modelo", "serie_id"])[["MASE", "RMSSE"]].mean().reset_index().assign(h="1-15")
    allh = pd.concat([per_h.assign(h=per_h["h"].astype(str)), ps], ignore_index=True)
    keep = [str(h) for h in (1, 3, 6, 9, 12, 15)] + ["1-15"]
    t = allh[allh["h"].isin(keep)].groupby(["modelo", "h"])[["MASE", "RMSSE"]].agg(["mean", "median"])
    t.columns = [f"{a}_{'media' if b == 'mean' else 'mediana'}" for a, b in t.columns]
    return t.reset_index(), allh


def wide(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    order = ["1", "3", "6", "9", "12", "15", "1-15"]
    out = {}
    for m, g in table.groupby("modelo"):
        g = g.set_index("h")
        out[m] = {f"h={h}": f"{g.loc[h, metric + '_media']:.2f}/{g.loc[h, metric + '_mediana']:.2f}" for h in order}
    return pd.DataFrame(out).T


def run_group(gname: str, Y: np.ndarray, names: list[str]) -> pd.DataFrame:
    log = open(LOGS / f"fid_{gname}_log.txt", "w", encoding="utf-8")

    def p(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()

    t0 = time.time()
    variants = GROUPS[gname]
    p(f"GRUPO {gname}: {list(variants)}")
    tasks = [(i, names[i], Y[i], variants) for i in range(Y.shape[0])]
    rows, infos = [], []
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        for r, inf in pool.imap_unordered(run_one, tasks):
            rows += r
            infos += inf
    raw = pd.DataFrame(rows).sort_values(["serie_id", "origen", "modelo", "h"])
    raw.to_csv(RES / f"fid_crudos_{gname}.csv", index=False)
    info = pd.DataFrame(infos)
    info.to_csv(RES / f"fid_ordenes_{gname}.csv", index=False)
    table, allh = summarize(raw)
    allh.to_csv(RES / f"fid_metricas_por_serie_{gname}.csv", index=False)
    for metric in ("MASE", "RMSSE"):
        p(f"\n{metric} media/mediana (paper FTGM 1-15: {PAPER_FTGM[metric]['1-15']})")
        p(wide(table, metric).to_string())
    for v, g in info.groupby("modelo"):
        p(f"{v}: N -> {g['N'].value_counts().sort_index().to_dict()}; sustituciones SNaive {int((g['fallo'] != '').sum())}")
    p(f"tiempo {time.time() - t0:.1f} s")
    log.close()
    return table.assign(grupo=gname)


def load_series() -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(C.RESULTS / "exp1_series_mensuales.csv", index_col=0)
    assert df.shape == (63, 70)
    return df.to_numpy().T.astype(float), list(df.columns)


def main(groups: list[str]) -> None:
    Y, names = load_series()
    tabs = [run_group(g, Y, names) for g in groups]
    new = pd.concat(tabs)
    path = RES / "fid_resumen.csv"
    if path.exists():
        old = pd.read_csv(path, dtype={"h": str})
        old = old[~old["grupo"].isin(groups)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or list(GROUPS))
