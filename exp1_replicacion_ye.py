"""Experimento 1 - Replicacion de fidelidad (RQ1) del FTGM de Ye, Xie, Boylan & Shang (2024).

Protocolo (paper, secciones 4.1-4.5):
  * 70 series departamento-tienda del M5, agregadas a meses calendario completos
    (feb-2011 .. abr-2016 = 63 observaciones; se descartan ene-2011 y may-2016 parciales).
  * Evaluacion rolling-origin expansiva: in-sample inicial 43, horizonte 15, 6 origenes
    (43..48).
  * Metricas MASE y RMSSE escaladas por el naive de un paso dentro de muestra (Ec. 33-34).
    Se promedian los 6 origenes por serie y luego se toma media y mediana sobre las 70.

Modelos:
  FTGM_OLS_paper  : Algoritmo 1 (validacion = ultimos 6 del in-sample, N in 1..5, lambda=0);
                    parametros estimados con el tramo de entrenamiento (o-6 obs) y pronostico
                    por integracion de la EDO hasta t = o+15 (lectura literal del paper,
                    confirmada por la Tabla C.1, ver exp1_tabla_c1.csv).
  FTGM_OLS_refit  : igual, pero re-estimando con todo el in-sample (o obs) el orden elegido.
  FTGM_Ridge      : Algoritmo 1 del motor con rejilla ridge (0, 0.5, 3, 10), re-estimado con o obs.
  FTGM_OLS_ancla  : DIAGNOSTICO (no es del paper): como FTGM_OLS_refit, pero la EDO continua desde
                    la suma acumulada observada en el origen.
  Motor_torneo    : ForecastService de produccion (modo legado period=12, horizon=15).
  SNaive, Naive, ETS (auto AICc entre 9 especificaciones), SARIMA (rejilla por AICc, D=1).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools
import sys
import time
import warnings
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common as C
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from statsmodels.tsa.stattools import kpss  # noqa: E402

warnings.filterwarnings("ignore")

H = 15
ORIGINS = list(range(43, 49))
VAL = 6
T = 12
REPORT_H = [1, 3, 6, 9, 12, 15]
DEPTS = ["HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2", "FOODS_1", "FOODS_2", "FOODS_3"]
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
RIDGE_GRID = (0.0, 0.5, 3.0, 10.0)
MODELS = [
    "FTGM_OLS_paper", "FTGM_OLS_refit", "FTGM_Ridge", "FTGM_OLS_ancla", "Motor_torneo",
    "SNaive", "Naive", "ETS", "SARIMA",
]

# Valores FTGM y SNaive de la Tabla 1 del paper (media/mediana), extraidos del texto.
PAPER = {
    ("FTGM", "MASE"): {1: (1.09, 0.82), 3: (1.24, 1.00), 6: (1.30, 1.02), 9: (2.04, 1.23),
                       12: (2.02, 1.28), 15: (1.99, 1.20), "1-15": (1.65, 1.28)},
    ("FTGM", "RMSSE"): {1: (0.92, 0.73), 3: (1.05, 0.86), 6: (1.08, 0.82), 9: (1.65, 1.11),
                        12: (1.69, 0.99), 15: (1.67, 1.06), "1-15": (1.38, 1.04)},
    ("SNaive", "MASE"): {1: (2.12, 1.75), 3: (2.07, 1.71), 6: (2.29, 2.12), 9: (2.72, 2.03),
                         12: (3.75, 3.25), 15: (3.74, 3.02), "1-15": (2.69, 2.35)},
    ("SNaive", "RMSSE"): {1: (1.81, 1.53), 3: (1.75, 1.54), 6: (1.95, 1.85), 9: (2.26, 1.71),
                          12: (2.99, 2.50), 15: (3.01, 2.54), "1-15": (2.38, 2.04)},
    ("SARIMA", "MASE"): {1: (1.29, 1.14), 3: (1.60, 1.48), 6: (1.91, 1.60), 9: (2.64, 1.93),
                         12: (3.47, 2.54), 15: (3.62, 2.56), "1-15": (2.44, 1.98)},
    ("HWES", "MASE"): {1: (1.17, 1.03), 3: (1.57, 1.45), 6: (1.91, 1.74), 9: (2.76, 2.29),
                       12: (3.51, 2.79), 15: (3.59, 2.48), "1-15": (2.47, 1.97)},
    ("SARIMA", "RMSSE"): {1: (1.17, 1.01), 3: (1.42, 1.25), 6: (1.71, 1.39), 9: (2.24, 1.69),
                          12: (2.98, 2.13), 15: (3.09, 2.23), "1-15": (2.27, 1.74)},
    ("HWES", "RMSSE"): {1: (1.06, 0.94), 3: (1.38, 1.27), 6: (1.70, 1.58), 9: (2.44, 2.06),
                        12: (3.04, 2.35), 15: (3.14, 2.31), "1-15": (2.34, 1.83)},
}
PAPER_C1 = [0.0056, 10771.06, 0.0013, -0.0023, -826.69, 1135.56]


# ------------------------------------------------------------------------- datos
def build_series() -> tuple[list[str], np.ndarray, pd.DatetimeIndex]:
    meta, sales = C.load_m5()
    dates = C.day_dates()
    names, rows = [], []
    for store in STORES:
        for dept in DEPTS:
            idx = meta.index[(meta.dept_id == dept) & (meta.store_id == store)]
            m = pd.Series(sales[idx].sum(axis=0), index=dates).resample("MS").sum()
            m = m.iloc[1:-1]  # meses calendario completos feb-2011..abr-2016
            names.append(f"{dept.title()}_{store}")
            rows.append(m.values)
    Y = np.vstack(rows)
    assert Y.shape == (70, 63), Y.shape
    return names, Y, m.index


# ------------------------------------------------------------------------ modelos
def snaive(ins: np.ndarray, h: int) -> np.ndarray:
    n = ins.size
    return np.array([ins[n - T + ((k - 1) % T)] for k in range(1, h + 1)])


def _anchored_predict(model, ins: np.ndarray, h: int) -> np.ndarray:
    """DIAGNOSTICO (no es del paper): continua la EDO desde la suma acumulada OBSERVADA en
    el origen, Y(t_m) = y_m + x_1/2 - x_m/2 (misma cuadratura trapezoidal de la Ec. 10), en
    lugar de desde la trayectoria y(t_m) que la EDO trae integrada desde t_1."""
    from app.ftgm import fourier

    m = ins.size
    y_obs = float(np.sum(ins) + 0.5 * ins[0] - 0.5 * ins[-1])
    grid = np.arange(m, m + h + 1, dtype=np.float64)
    x_hat, _ = model._integrate(grid, model.theta_, fourier.angular_frequency(T), y_start=y_obs)
    point = x_hat[1:]
    model._assert_stable(point, reference_max=float(np.max(np.abs(ins))) or 1.0)
    return np.clip(point, 0.0, None)


def ftgm_forecast(
    ins: np.ndarray, *, ridges: tuple[float, ...], refit: bool, anchored: bool = False
) -> tuple[np.ndarray, dict]:
    from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order

    cfg = FTGMConfig(period=T)
    sel = select_order(ins, cfg, validation_size=VAL, max_order=5, ridges=ridges)
    ranking = sorted((v, k) for k, v in sel.scores.items() if np.isfinite(v))
    info = {"order": sel.best_order, "ridge": sel.best_ridge, "fallo": "", "orden_usado": None}
    # Orden elegido primero; si la integracion final diverge, el siguiente del ranking.
    candidates = [sel.best_order] + [k for _, k in ranking if k != sel.best_order]
    for k in candidates:
        ridge = sel.best_ridge
        try:
            if anchored:
                fc = _anchored_predict(FTGM(order=k, config=FTGMConfig(period=T, ridge=ridge)).fit(ins), ins, H)
            elif refit:
                fc = FTGM(order=k, config=FTGMConfig(period=T, ridge=ridge)).fit(ins).predict(H).point
            else:
                train = ins[: ins.size - VAL]
                fc = FTGM(order=k, config=FTGMConfig(period=T, ridge=ridge)).fit(train).predict(VAL + H).point[VAL:]
            info["orden_usado"] = k
            if k != sel.best_order:
                info["fallo"] = f"orden {sel.best_order} diverge; se usa {k}"
            return fc, info
        except (FTGMError, ValueError) as exc:
            info["fallo"] = f"orden {k}: {exc}"
            continue
    info["fallo"] = "todos los ordenes divergen; se sustituye por SNaive"
    return snaive(ins, H), info


def motor_forecast(ins: np.ndarray, month_dates: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
    from app.application.forecast_service import ForecastService
    from app.presentation.schemas import ForecastRequest, ObservationPoint, ProductSeries

    pts = [ObservationPoint(date=d.date(), demand=float(v)) for d, v in zip(month_dates[: ins.size], ins)]
    req = ForecastRequest(period=T, horizon=H, series=[ProductSeries(product_id="s", points=pts)])
    out = ForecastService().forecast(req).forecasts[0]
    fc = np.array([p.predicted_demand for p in out.points], dtype=float)
    return fc, {"order": out.order_selected, "ridge": None, "fallo": out.status, "orden_usado": out.model}


ETS_SPECS = [
    ("add", None, False, None), ("add", "add", False, None), ("add", "add", True, None),
    ("add", None, False, "add"), ("add", "add", False, "add"), ("add", "add", True, "add"),
    ("mul", None, False, "mul"), ("mul", "add", False, "mul"), ("mul", "add", True, "mul"),
]


#: Salvaguarda de plausibilidad para ETS/SARIMA: se descarta una especificacion cuyo
#: pronostico no es finito o supera en valor absoluto 5 veces el maximo historico (auto.arima
#: y ets de R descartan de forma analoga ajustes numericamente inestables).
_PLAUSIBLE = 5.0


def _plausible(fc: np.ndarray, ins: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(fc)) and np.max(np.abs(fc)) <= _PLAUSIBLE * np.max(np.abs(ins)))


def ets_forecast(ins: np.ndarray) -> tuple[np.ndarray, dict]:
    best = None
    rejected = 0
    for err, trend, damped, seas in ETS_SPECS:
        try:
            res = ETSModel(
                ins, error=err, trend=trend, damped_trend=damped, seasonal=seas,
                seasonal_periods=T if seas else None,
            ).fit(disp=False, maxiter=2000)
            aicc = float(res.aicc)
            if not np.isfinite(aicc):
                continue
            if not _plausible(np.asarray(res.forecast(H), dtype=float), ins):
                rejected += 1
                continue
            if best is None or aicc < best[0]:
                best = (aicc, res, (err, trend, damped, seas))
        except Exception:  # noqa: BLE001
            continue
    if best is None:
        return snaive(ins, H), {"order": None, "ridge": None, "fallo": "ETS fallo", "orden_usado": None}
    fc = np.asarray(best[1].forecast(H), dtype=float)
    e, t, d, s = best[2]
    spec = f"ETS({e[0].upper()},{'N' if t is None else ('Ad' if d else 'A')},{'N' if s is None else s[0].upper()})"
    return fc, {"order": None, "ridge": None, "fallo": f"descartadas={rejected}" if rejected else "",
                "orden_usado": spec}


def sarima_forecast(ins: np.ndarray) -> tuple[np.ndarray, dict]:
    sd = ins[T:] - ins[:-T]
    try:
        pval = kpss(sd, regression="c", nlags="auto")[1]
    except Exception:  # noqa: BLE001
        pval = 1.0
    d = 1 if pval < 0.05 else 0
    best = None
    rejected = 0
    for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
        try:
            res = SARIMAX(
                ins, order=(p, d, q), seasonal_order=(P, 1, Q, T), trend="c" if d == 0 else "n",
            ).fit(disp=False, maxiter=200)
            aicc = float(res.aicc)
            if not np.isfinite(aicc):
                continue
            if not _plausible(np.asarray(res.forecast(H), dtype=float), ins):
                rejected += 1
                continue
            if best is None or aicc < best[0]:
                best = (aicc, res, (p, d, q, P, Q))
        except Exception:  # noqa: BLE001
            continue
    if best is None:
        return snaive(ins, H), {"order": None, "ridge": None, "fallo": "SARIMA fallo", "orden_usado": None}
    fc = np.asarray(best[1].forecast(H), dtype=float)
    p, d, q, P, Q = best[2]
    return fc, {"order": None, "ridge": None, "fallo": f"descartadas={rejected}" if rejected else "",
                "orden_usado": f"({p},{d},{q})({P},1,{Q})[12]"}


def run_series(args: tuple[int, str, np.ndarray, pd.DatetimeIndex]) -> tuple[list[dict], list[dict]]:
    warnings.filterwarnings("ignore")
    i, name, y, month_dates = args
    rows, infos = [], []
    for o in ORIGINS:
        ins, act = y[:o], y[o : o + H]
        s_mae, s_mse = C.paper_scales(ins)
        fcs: dict[str, tuple[np.ndarray, dict]] = {}
        t0 = time.perf_counter()
        fcs["FTGM_OLS_paper"] = ftgm_forecast(ins, ridges=(0.0,), refit=False)
        fcs["FTGM_OLS_refit"] = ftgm_forecast(ins, ridges=(0.0,), refit=True)
        fcs["FTGM_Ridge"] = ftgm_forecast(ins, ridges=RIDGE_GRID, refit=True)
        fcs["FTGM_OLS_ancla"] = ftgm_forecast(ins, ridges=(0.0,), refit=True, anchored=True)
        fcs["Motor_torneo"] = motor_forecast(ins, month_dates)
        fcs["SNaive"] = (snaive(ins, H), {})
        fcs["Naive"] = (np.full(H, ins[-1]), {})
        fcs["ETS"] = ets_forecast(ins)
        fcs["SARIMA"] = sarima_forecast(ins)
        for model, (fc, info) in fcs.items():
            fc = np.asarray(fc, dtype=float)
            for h in range(1, H + 1):
                rows.append({
                    "serie_id": i + 1, "serie": name, "origen": o, "modelo": model, "h": h,
                    "real": act[h - 1], "pronostico": fc[h - 1], "escala_mae": s_mae, "escala_mse": s_mse,
                })
            if info:
                infos.append({"serie_id": i + 1, "serie": name, "origen": o, "modelo": model, **info})
        infos.append({"serie_id": i + 1, "serie": name, "origen": o, "modelo": "_tiempo_s",
                      "order": None, "ridge": None, "fallo": f"{time.perf_counter() - t0:.2f}", "orden_usado": None})
    return rows, infos


# ---------------------------------------------------------------------- metricas
def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["ae"] = (raw["real"] - raw["pronostico"]).abs()
    raw["se"] = (raw["real"] - raw["pronostico"]) ** 2
    raw["MASE"] = raw["ae"] / raw["escala_mae"]
    raw["RMSSE"] = np.sqrt(raw["se"] / raw["escala_mse"])
    # Por horizonte: promedio de los 6 origenes por serie.
    per_h = raw.groupby(["modelo", "serie_id", "h"])[["MASE", "RMSSE"]].mean().reset_index()
    # 1-15: metrica sobre los 15 pasos de cada origen, luego promedio de origenes.
    g = raw.groupby(["modelo", "serie_id", "origen"])
    o = pd.DataFrame({
        "MASE": g["ae"].mean() / g["escala_mae"].first(),
        "RMSSE": np.sqrt(g["se"].mean() / g["escala_mse"].first()),
    }).reset_index()
    per_series = o.groupby(["modelo", "serie_id"])[["MASE", "RMSSE"]].mean().reset_index()
    per_series["h"] = "1-15"
    allh = pd.concat([per_h.assign(h=per_h["h"].astype(str)), per_series], ignore_index=True)
    table = (
        allh[allh["h"].isin([str(h) for h in REPORT_H] + ["1-15"])]
        .groupby(["modelo", "h"])[["MASE", "RMSSE"]]
        .agg(["mean", "median"])
    )
    table.columns = [f"{a}_{'media' if b == 'mean' else 'mediana'}" for a, b in table.columns]
    table = table.reset_index()
    # Variante: media/mediana sobre el conjunto agrupado de 70 x 6 = 420 valores.
    pooled_h = raw.groupby(["modelo", "h"])[["MASE", "RMSSE"]].agg(["mean", "median"])
    pooled_h.columns = [f"{a}_{'media' if b == 'mean' else 'mediana'}" for a, b in pooled_h.columns]
    pooled_h = pooled_h.reset_index()
    pooled_h["h"] = pooled_h["h"].astype(str)
    pooled_o = o.groupby("modelo")[["MASE", "RMSSE"]].agg(["mean", "median"])
    pooled_o.columns = [f"{a}_{'media' if b == 'mean' else 'mediana'}" for a, b in pooled_o.columns]
    pooled_o = pooled_o.reset_index().assign(h="1-15")
    pooled = pd.concat([pooled_h[pooled_h["h"].isin([str(h) for h in REPORT_H])], pooled_o], ignore_index=True)
    return table, allh, pooled


def main() -> None:
    log_path = C.LOGS / "exp1_log.txt"
    sys.stdout = C.Tee(log_path)  # type: ignore[assignment]
    t_start = time.time()
    print("EXPERIMENTO 1 - Replicacion de fidelidad (Ye et al., 2024)")
    print(C.environment())
    names, Y, month_dates = build_series()
    print(f"Series: {Y.shape[0]} x {Y.shape[1]} meses ({month_dates[0].date()} .. {month_dates[-1].date()})")
    pd.DataFrame(Y.T, index=month_dates.date, columns=names).to_csv(C.RESULTS / "exp1_series_mensuales.csv")

    # Verificacion de la construccion mensual contra la Tabla C.1 (Hobbies_1_CA_1, N=1).
    from app.ftgm import FTGM, FTGMConfig

    y0 = Y[0]
    c1 = []
    for n in (37, 42, 43):
        th = FTGM(order=1, config=FTGMConfig(period=T)).fit(y0[:n]).theta_
        c1.append({"n_obs_estimacion": n, **{f"theta_{j}": float(v) for j, v in enumerate(th)}})
    c1.append({"n_obs_estimacion": "paper_TablaC1", **{f"theta_{j}": v for j, v in enumerate(PAPER_C1)}})
    pd.DataFrame(c1).to_csv(C.RESULTS / "exp1_tabla_c1.csv", index=False)
    print("Tabla C.1 (theta = [a0, b0, a11, a21, b11, b21]) Hobbies_1_CA_1, N=1:")
    print(pd.DataFrame(c1).to_string(index=False))

    # Verificacion numerica de la integracion de la EDO del motor (RK45, rtol 1e-6) contra
    # DOP853 con rtol 1e-11, para N = 1..5, en el entrenamiento del primer origen (37 obs).
    from scipy.integrate import solve_ivp

    from app.ftgm import FTGMError, fourier

    chk = []
    w = fourier.angular_frequency(T)
    for i in range(Y.shape[0]):
        ytr = Y[i][:37]
        for N in range(1, 6):
            try:
                mdl = FTGM(order=N, config=FTGMConfig(period=T)).fit(ytr)
                fc = mdl.predict(21).point
            except FTGMError:
                chk.append({"serie_id": i + 1, "N": N, "estable": False})
                continue
            tt = np.arange(1, 59, dtype=float)
            a, b = fourier.evaluate_parameters(tt, mdl.theta_, N, w)
            th = mdl.theta_

            def rhs(t_, y_, th=th, N=N):
                aa, bb = fourier.evaluate_parameters(np.array([t_]), th, N, w)
                return [aa[0] * y_[0] + bb[0]]

            sol = solve_ivp(rhs, (1.0, 58.0), [b[0] / (1 - a[0])], t_eval=tt, method="DOP853",
                            rtol=1e-11, atol=1e-9)
            ref = a * sol.y[0] + b
            chk.append({"serie_id": i + 1, "N": N, "estable": True,
                        "max_dif_rel_ajuste": float(np.max(np.abs(ref[:37] - mdl.fitted_)) / np.mean(ytr)),
                        "max_dif_rel_pronostico": float(np.max(np.abs(np.clip(ref[37:], 0, None) - fc))
                                                        / np.mean(ytr))})
    chk_df = pd.DataFrame(chk)
    chk_df.to_csv(C.RESULTS / "exp1_verificacion_edo.csv", index=False)
    ok = chk_df[chk_df["estable"]]
    print(f"Verificacion EDO (motor RK45 vs DOP853 rtol=1e-11): {len(ok)} ajustes estables de {len(chk_df)}; "
          f"max diferencia relativa al nivel medio: ajuste {ok['max_dif_rel_ajuste'].max():.2e}, "
          f"pronostico {ok['max_dif_rel_pronostico'].max():.2e}")

    tasks = [(i, names[i], Y[i], month_dates) for i in range(Y.shape[0])]
    rows, infos = [], []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for k, (r, inf) in enumerate(pool.imap_unordered(run_series, tasks), 1):
            rows += r
            infos += inf
            if k % 10 == 0:
                print(f"  {k}/70 series ({time.time() - t_start:.0f} s)")
    raw = pd.DataFrame(rows).sort_values(["serie_id", "origen", "modelo", "h"])
    raw.to_csv(C.RESULTS / "exp1_pronosticos_crudos.csv", index=False)
    info = pd.DataFrame(infos).sort_values(["serie_id", "origen", "modelo"])
    info.to_csv(C.RESULTS / "exp1_modelos_elegidos.csv", index=False)

    table, allh, pooled = summarize(raw)
    allh.to_csv(C.RESULTS / "exp1_metricas_por_serie.csv", index=False)
    table.to_csv(C.RESULTS / "exp1_tabla_resumen.csv", index=False)
    pooled.to_csv(C.RESULTS / "exp1_tabla_resumen_agrupada420.csv", index=False)

    order_h = {str(h): i for i, h in enumerate(REPORT_H)} | {"1-15": 99}
    for metric in ("MASE", "RMSSE"):
        piv = table.assign(o=table["h"].map(order_h)).sort_values("o")
        print(f"\n{metric} (media / mediana sobre 70 series; promedio de 6 origenes por serie)")
        wide = piv.pivot_table(index="modelo", columns="o", values=[f"{metric}_media", f"{metric}_mediana"])
        out = pd.DataFrame(index=wide.index)
        for oh, label in sorted((v, k) for k, v in order_h.items()):
            out[f"h={label}"] = [
                f"{wide.loc[m, (f'{metric}_media', oh)]:.2f}/{wide.loc[m, (f'{metric}_mediana', oh)]:.2f}"
                for m in wide.index
            ]
        print(out.loc[[m for m in MODELS if m in out.index]].to_string())

    # Brecha contra el paper.
    gap = []
    for (model_p, metric), vals in PAPER.items():
        ours_models = {"FTGM": ["FTGM_OLS_paper", "FTGM_OLS_refit", "FTGM_Ridge"], "SNaive": ["SNaive"],
                       "SARIMA": ["SARIMA"], "HWES": ["ETS"]}[model_p]
        for ours in ours_models:
            for h, (pm, pmed) in vals.items():
                r = table[(table["modelo"] == ours) & (table["h"] == str(h))].iloc[0]
                gap.append({
                    "modelo_paper": model_p, "modelo_replica": ours, "metrica": metric, "h": str(h),
                    "paper_media": pm, "replica_media": round(r[f"{metric}_media"], 4),
                    "brecha_media": round(r[f"{metric}_media"] - pm, 4),
                    "paper_mediana": pmed, "replica_mediana": round(r[f"{metric}_mediana"], 4),
                    "brecha_mediana": round(r[f"{metric}_mediana"] - pmed, 4),
                })
    gap_df = pd.DataFrame(gap)
    gap_df.to_csv(C.RESULTS / "exp1_brecha_vs_paper.csv", index=False)
    print("\nBrecha vs Tabla 1 del paper (replica - paper), h=1-15:")
    print(gap_df[gap_df["h"] == "1-15"].to_string(index=False))

    # % de series donde cada FTGM supera a SNaive (RMSSE y MASE promedio 1-15).
    s15 = allh[allh["h"] == "1-15"].pivot(index="serie_id", columns="modelo", values=["MASE", "RMSSE"])
    from scipy.stats import wilcoxon

    wins = []
    for model in MODELS:
        if model == "SNaive":
            continue
        for metric in ("MASE", "RMSSE"):
            a, b = s15[(metric, model)], s15[(metric, "SNaive")]
            stat = wilcoxon(a, b)
            wins.append({"modelo": model, "metrica": metric, "pct_series_mejor_que_SNaive": round(100 * float((a < b).mean()), 1),
                         "n_series": int(a.size), "wilcoxon_p": float(stat.pvalue)})
    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(C.RESULTS / "exp1_victorias_vs_snaive.csv", index=False)
    print("\n% series mejor que SNaive (1-15) y Wilcoxon pareado:")
    print(wins_df.to_string(index=False))

    # Ranking: veces en primer lugar por RMSSE 1-15 (comparable a la Fig. 4a del paper).
    cmp = [m for m in MODELS if m not in ("FTGM_OLS_refit", "FTGM_Ridge", "FTGM_OLS_ancla", "Motor_torneo")]
    first = s15["RMSSE"][cmp].idxmin(axis=1).value_counts()
    rank_df = first.rename("veces_primer_lugar").reset_index().rename(columns={"index": "modelo"})
    rank_df.to_csv(C.RESULTS / "exp1_primer_lugar_rmsse.csv", index=False)
    print("\nPrimer lugar por RMSSE 1-15 (FTGM_OLS_paper vs SNaive, Naive, ETS, SARIMA):")
    print(rank_df.to_string(index=False))

    # Ordenes elegidos y fallos del FTGM.
    for model in ("FTGM_OLS_paper", "FTGM_OLS_refit", "FTGM_Ridge", "FTGM_OLS_ancla"):
        sub = info[info["modelo"] == model]
        print(f"\n{model}: orden elegido -> {sub['order'].value_counts().sort_index().to_dict()}; "
              f"ridge -> {sub['ridge'].value_counts().sort_index().to_dict()}; "
              f"casos con divergencia/sustitucion: {(sub['fallo'] != '').sum()} de {len(sub)}")
    sub = info[info["modelo"] == "Motor_torneo"]
    print(f"\nMotor_torneo: ganador -> {sub['orden_usado'].value_counts().to_dict()}")
    for model in ("ETS", "SARIMA"):
        sub = info[info["modelo"] == model]
        print(f"{model}: origenes con especificaciones descartadas por la salvaguarda: "
              f"{(sub['fallo'].fillna('') != '').sum()} de {len(sub)}")
    sub = info[info["modelo"] == "ETS"]
    print(f"ETS elegido -> {sub['orden_usado'].value_counts().to_dict()}")
    sub = info[info["modelo"] == "SARIMA"]
    n_d1 = int(sub["orden_usado"].fillna("").str.contains(r"^\(\d,1,", regex=True).sum())
    print(f"SARIMA: d=1 en {n_d1} de {len(sub)}; "
          f"top: {sub['orden_usado'].value_counts().head(5).to_dict()}")
    print(f"\nTiempo total: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
