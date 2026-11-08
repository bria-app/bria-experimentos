"""Experimento 3 - Series cortas tipo MYPE (RQ2) con el pipeline de PRODUCCION del motor.

Datos: 300 series producto-tienda del M5 (muestreo estratificado proporcional por categoria,
semilla 42) con venta media diaria >= 0.2 en los 24 meses previos al corte y con al menos una
venta antes del inicio de la ventana de 36 meses (para que todas las historias sean reales).

Protocolo:
  * Corte comun as_of = 2016-02-01 (lunes y primer dia de mes). Historias truncadas a
    L = 6, 9, 12, 18, 24 meses (y 36 como referencia adicional) que terminan el 2016-01-31.
  * Puntos diarios dispersos (solo dias con venta), como los arma la API
    (inventory-dss-api, erp_analysis.build_points).
  * ForecastService().forecast(ForecastRequest(frequency="auto", horizon_days=H, as_of=as_of))
    con H = 30 y 90, exactamente el contrato que usa la API.
  * Evaluacion sobre los periodos (meses o semanas) que devuelve el motor: el real de cada
    periodo es la suma de ventas diarias del M5 en [inicio, inicio siguiente).
  * Referencias puras sobre los MISMOS periodos y la misma historia cruda (sin reparar ni
    limpiar): Naive (ultimo periodo), SNaive (mismo periodo de la temporada anterior si la
    historia lo permite; si no, Naive) y, en modo mensual (L >= 24), FTGM solo (Algoritmo 1
    con rejilla ridge del motor, sin torneo).

Sub-experimento de censura CONTROLADA (inyectada sobre datos M5): en 100 de los 300 productos
(L = 12 y 24 meses) se ponen en cero L/4 ventanas aleatorias de 7 a 14 dias de la historia.
Condiciones: A original sin censura; B censurada sin marcar (los ceros se toman como demanda);
C censurada y marcada como quiebre (stockout_flag = True en esos dias). Se evalua contra el
futuro real (no censurado).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import warnings
from datetime import date, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common as C

warnings.filterwarnings("ignore")

AS_OF = date(2016, 2, 1)
LENGTHS = [6, 9, 12, 18, 24, 36]
HORIZONS = [30, 90]
N_SAMPLE = 300
N_CENSOR = 100
CENSOR_LENGTHS = [12, 24]
MIN_MEAN = 0.2
RIDGE_GRID = (0.0, 0.5, 3.0, 10.0)
FAMILY = {
    "FTGM": "Familia FTGM", "FTGMCombo": "Familia FTGM", "CrostonSBA": "Croston-SBA",
    "MovingAverage": "Promedio movil", "DampedTrend": "Suavizado amortiguado",
    "SeasonalDamped": "Suavizado estacional", "SeasonalNaive": "Naive estacional", "None": "Omitido",
}


def day_index(d: date) -> int:
    return (d - C.M5_START).days


# -------------------------------------------------------------------- muestra
def sample_products() -> tuple[pd.DataFrame, np.ndarray]:
    meta, sales = C.load_m5()
    i_asof = day_index(AS_OF)
    i24 = day_index(C.add_months(AS_OF, -24))
    i36 = day_index(C.add_months(AS_OF, -36))
    mean24 = sales[:, i24:i_asof].mean(axis=1)
    first_sale = np.argmax(sales > 0, axis=1)
    has_sale = (sales > 0).any(axis=1)
    elig = (mean24 >= MIN_MEAN) & has_sale & (first_sale <= i36)
    pool = meta[elig].copy()
    pool["mean24"] = mean24[elig]
    rng = np.random.default_rng(C.SEED)
    counts = pool["cat_id"].value_counts().sort_index()
    alloc = (counts / counts.sum() * N_SAMPLE).round().astype(int)
    alloc.iloc[np.argmax(alloc.values)] += N_SAMPLE - alloc.sum()
    picked = []
    for cat, k in alloc.items():
        idx = pool.index[pool["cat_id"] == cat].to_numpy()
        picked += list(rng.choice(idx, size=int(k), replace=False))
    sample = pool.loc[sorted(picked)].copy()
    sample["row"] = sample.index
    print(f"Elegibles: {int(elig.sum())} de {len(meta)}; poblacion por categoria: {counts.to_dict()}; "
          f"muestra: {alloc.to_dict()}")
    return sample.reset_index(drop=True), sales[sample["row"].to_numpy()]


# ---------------------------------------------------------------- utilidades
def make_points(daily: np.ndarray, start: date, flags: np.ndarray | None = None) -> list:
    from app.presentation.schemas import ObservationPoint

    pts = []
    for k, v in enumerate(daily):
        flag = bool(flags[k]) if flags is not None else False
        if v > 0 or flag:
            pts.append(ObservationPoint(date=start + timedelta(days=k), demand=float(v), stockout_flag=flag))
    return pts


def actual_for(full: np.ndarray, starts: list[date], period: int) -> np.ndarray:
    from app.ftgm import preprocessing as prep

    out = []
    for s in starts:
        e = prep.next_bucket(s, period)
        out.append(float(full[day_index(s) : day_index(e)].sum()))
    return np.array(out)


def run_engine(points: list, horizon_days: int) -> object:
    from app.application.forecast_service import ForecastService
    from app.presentation.schemas import ForecastRequest, ProductSeries

    req = ForecastRequest(
        frequency="auto", horizon_days=horizon_days, as_of=AS_OF,
        series=[ProductSeries(product_id="p", points=points)],
    )
    return ForecastService().forecast(req).forecasts[0]


def baselines(points: list, period: int, h: int, monthly: bool) -> dict[str, tuple[np.ndarray, str]]:
    from app.baselines import seasonal_naive_forecast
    from app.ftgm import FTGM, FTGMConfig, FTGMError, select_order
    from app.ftgm import preprocessing as prep

    b = prep.bucketize([p.date for p in points], np.array([p.demand for p in points]),
                       np.zeros(len(points), dtype=bool), period, AS_OF)
    x = b.demand
    out = {"Naive": (np.full(h, x[-1]), ""), "SNaive": (seasonal_naive_forecast(x, h, period), "")}
    if monthly:
        note = ""
        try:
            cap = min(5, max(1, int(round(2.0 * x.size / period))))
            sel = select_order(x, FTGMConfig(period=period), max_order=cap, ridges=RIDGE_GRID)
            fc = FTGM(order=sel.best_order, config=FTGMConfig(period=period, ridge=sel.best_ridge)).fit(x).predict(h).point
        except FTGMError as exc:
            fc, note = seasonal_naive_forecast(x, h, period), f"FTGM fallo ({exc}); se usa SNaive"
        out["FTGM_solo"] = (np.clip(fc, 0, None), note)
    return out


# ------------------------------------------------------------------- trabajo
def work(args: tuple) -> tuple[list[dict], list[dict], list[dict]]:
    warnings.filterwarnings("ignore")
    pid, item, cat, full, do_censor = args
    rows, periods, cens = [], [], []
    for L in LENGTHS:
        start = C.add_months(AS_OF, -L)
        hist = full[day_index(start) : day_index(AS_OF)]
        pts = make_points(hist, start)
        for H in HORIZONS:
            t0 = time.perf_counter()
            fc = run_engine(pts, H)
            dt = time.perf_counter() - t0
            if fc.status == "skipped" or not fc.points:
                rows.append({"producto": pid, "item": item, "cat": cat, "L_meses": L, "H_dias": H,
                             "modelo_motor": fc.model, "status": fc.status, "nota": fc.fallback_reason})
                continue
            starts = [p.date for p in fc.points]
            act = actual_for(full, starts, fc.period)
            pred = {"Motor": (np.array([p.predicted_demand for p in fc.points]), "")}
            pred |= baselines(pts, fc.period, len(starts), fc.period == 12)
            for model, (f, note) in pred.items():
                rows.append({
                    "producto": pid, "item": item, "cat": cat, "L_meses": L, "H_dias": H, "metodo": model,
                    "modelo_motor": fc.model, "status": fc.status, "frecuencia": fc.frequency,
                    "periodo": fc.period, "n_periodos_hist": fc.diagnostics.n_periods,
                    "intermitente": fc.diagnostics.intermittent, "n_periodos_fc": len(starts),
                    "dias_eval": (C.add_months(starts[-1], 1) if fc.period == 12 else starts[-1] + timedelta(7)).toordinal()
                    - starts[0].toordinal(),
                    "real_total": act.sum(), "pron_total": float(np.sum(f)),
                    "ea_total": abs(float(np.sum(f)) - act.sum()),
                    "ea_periodos": float(np.abs(f - act).sum()), "real_periodos": act.sum(),
                    "eac_periodos": float(((f - act) ** 2).sum()),
                    "seg_motor": dt if model == "Motor" else None, "nota": note,
                })
                for s, a, v in zip(starts, act, f):
                    periods.append({"producto": pid, "L_meses": L, "H_dias": H, "metodo": model,
                                    "inicio_periodo": s.isoformat(), "real": a, "pronostico": float(v)})
    if do_censor:
        rng = np.random.default_rng(C.SEED + pid)
        for L in CENSOR_LENGTHS:
            start = C.add_months(AS_OF, -L)
            hist = full[day_index(start) : day_index(AS_OF)].copy()
            mask = np.zeros(hist.size, dtype=bool)
            n_win = L // 4
            tries = 0
            while mask.sum() == 0 or (tries < 200 and _n_windows(mask) < n_win):
                tries += 1
                ln = int(rng.integers(7, 15))
                s0 = int(rng.integers(0, hist.size - ln))
                if mask[max(0, s0 - 7) : s0 + ln + 7].any():
                    continue
                mask[s0 : s0 + ln] = True
            cens_hist = hist.copy()
            cens_hist[mask] = 0.0
            lost = float(hist[mask].sum())
            conds = {
                "A_original": make_points(hist, start),
                "B_censurada_sin_marcar": make_points(cens_hist, start),
                "C_censurada_marcada": make_points(cens_hist, start, flags=mask),
            }
            for H in HORIZONS:
                hists: dict[str, dict] = {}
                block: list[dict] = []
                for cond, pts in conds.items():
                    fc = run_engine(pts, H)
                    if not fc.points:
                        block.append({"producto": pid, "L_meses": L, "H_dias": H, "condicion": cond,
                                      "status": fc.status})
                        continue
                    hists[cond] = {hp.date: hp.cleaned for hp in fc.history}
                    starts = [p.date for p in fc.points]
                    act = actual_for(full, starts, fc.period)
                    f = np.array([p.predicted_demand for p in fc.points])
                    hist_obs = sum(hp.observed for hp in fc.history)
                    hist_clean = sum(hp.cleaned for hp in fc.history)
                    block.append({
                        "producto": pid, "L_meses": L, "H_dias": H, "condicion": cond, "status": fc.status,
                        "modelo_motor": fc.model, "periodo": fc.period, "dias_censurados": int(mask.sum()),
                        "ventanas": _n_windows(mask), "unidades_censuradas": lost,
                        "periodos_quiebre": fc.diagnostics.stockout_periods,
                        "periodos_imputados": fc.diagnostics.imputed_periods,
                        "hist_observada": hist_obs, "hist_reparada": hist_clean,
                        "real_total": act.sum(), "pron_total": float(f.sum()),
                        "ea_total": abs(float(f.sum()) - act.sum()), "ea_periodos": float(np.abs(f - act).sum()),
                    })
                # Distorsion de la historia que ve el modelo (serie limpiada/reparada) respecto
                # de la historia limpiada SIN censura (condicion A), en los periodos comunes.
                ref = hists.get("A_original")
                for r in block:
                    hx = hists.get(r["condicion"])
                    if ref is None or hx is None:
                        continue
                    common = [d for d in ref if d in hx]
                    den = sum(ref[d] for d in common)
                    r["distorsion_historia_pct"] = (
                        100.0 * sum(abs(hx[d] - ref[d]) for d in common) / den if den > 0 else None
                    )
                cens += block
    return rows, periods, cens


def _n_windows(mask: np.ndarray) -> int:
    return int(np.sum(np.diff(np.concatenate([[0], mask.astype(int)])) == 1))


# ------------------------------------------------------------------- resumen
def wape(num: pd.Series, den: pd.Series) -> float:
    return float(num.sum() / den.sum() * 100.0)


def main() -> None:
    sys.stdout = C.Tee(C.LOGS / "exp3_log.txt")  # type: ignore[assignment]
    t_start = time.time()
    print("EXPERIMENTO 3 - Series cortas tipo MYPE con el pipeline de produccion")
    print(C.environment())
    sample, S = sample_products()
    sample.drop(columns=["row"]).to_csv(C.RESULTS / "exp3_muestra.csv", index=False)
    rng = np.random.default_rng(C.SEED)
    censor_ids = set(rng.choice(len(sample), size=N_CENSOR, replace=False).tolist())
    tasks = [(i, sample.loc[i, "id"], sample.loc[i, "cat_id"], S[i], i in censor_ids) for i in range(len(sample))]
    rows, periods, cens = [], [], []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for k, (r, p, c) in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            rows += r
            periods += p
            cens += c
            if k % 25 == 0:
                print(f"  {k}/{len(tasks)} productos ({time.time() - t_start:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "exp3_resultados_por_producto.csv", index=False)
    pd.DataFrame(periods).to_csv(C.RESULTS / "exp3_pronosticos_por_periodo.csv", index=False)
    cdf = pd.DataFrame(cens)
    cdf.to_csv(C.RESULTS / "exp3_censura_por_producto.csv", index=False)

    skipped = df[df.get("metodo").isna()] if "metodo" in df else df.iloc[0:0]
    print(f"\nCasos omitidos por el motor (status skipped): {len(skipped)}")
    df = df[df["metodo"].notna()]

    # 1) Ganador del torneo por largo de historia.
    motor = df[df["metodo"] == "Motor"].copy()
    motor["familia"] = motor["modelo_motor"].map(lambda m: FAMILY.get(m, m))
    share = (motor.groupby(["H_dias", "L_meses"])["familia"].value_counts(normalize=True) * 100).round(1)
    share = share.rename("pct").reset_index()
    share.to_csv(C.RESULTS / "exp3_ganador_por_largo.csv", index=False)
    freq = motor.groupby(["H_dias", "L_meses"])["frecuencia"].value_counts().rename("n").reset_index()
    freq.to_csv(C.RESULTS / "exp3_frecuencia_por_largo.csv", index=False)
    print("\nGanador del motor (% de productos) por largo de historia:")
    print(share.pivot_table(index=["H_dias", "L_meses"], columns="familia", values="pct", fill_value=0).to_string())
    print("\nFrecuencia elegida por el motor:")
    print(freq.pivot_table(index=["H_dias", "L_meses"], columns="frecuencia", values="n", fill_value=0).to_string())

    # 2) Error del pipeline vs referencias.
    from scipy.stats import wilcoxon

    summ, tests = [], []
    for (H, L), g in df.groupby(["H_dias", "L_meses"]):
        base = g[g["metodo"] == "Motor"].set_index("producto")
        for m, gm in g.groupby("metodo"):
            gm = gm.set_index("producto")
            summ.append({
                "H_dias": H, "L_meses": L, "metodo": m, "n_productos": len(gm),
                "WAPE_total_pct": round(wape(gm["ea_total"], gm["real_total"]), 2),
                "WAPE_periodos_pct": round(wape(gm["ea_periodos"], gm["real_periodos"]), 2),
                "MAE_total_unid": round(float(gm["ea_total"].mean()), 3),
                "MAE_periodo_unid": round(float(gm["ea_periodos"].sum() / gm["n_periodos_fc"].sum()), 3),
                # Metricas robustas: mediana del error porcentual absoluto del total por producto
                # y % de productos con sobrepronostico extremo (total pronosticado > 3x el real).
                "MdAPE_total_pct": round(float(np.median(
                    (gm["ea_total"] / gm["real_total"])[gm["real_total"] > 0]) * 100), 2),
                "pct_sobrepronostico_x3": round(float(
                    (gm["pron_total"] > 3 * gm["real_total"]).mean() * 100), 1),
                "fallos_FTGM_solo": int((gm["nota"].fillna("") != "").sum()) if m == "FTGM_solo" else None,
            })
            if m == "Motor":
                continue
            common_idx = base.index.intersection(gm.index)
            a = base.loc[common_idx, "ea_periodos"] / base.loc[common_idx, "n_periodos_fc"]
            b = gm.loc[common_idx, "ea_periodos"] / gm.loc[common_idx, "n_periodos_fc"]
            at, bt = base.loc[common_idx, "ea_total"], gm.loc[common_idx, "ea_total"]
            d = a - b
            try:
                pv = float(wilcoxon(a, b).pvalue)
            except ValueError:
                pv = float("nan")
            try:
                pvt = float(wilcoxon(at, bt).pvalue)
            except ValueError:
                pvt = float("nan")
            tests.append({
                "H_dias": H, "L_meses": L, "comparacion": f"Motor vs {m}", "n": len(common_idx),
                "motor_gana_pct": round(100 * float((d < 0).mean()), 1),
                "empate_pct": round(100 * float((d == 0).mean()), 1),
                "motor_pierde_pct": round(100 * float((d > 0).mean()), 1),
                "mediana_dif_MAE_periodo": round(float(np.median(d)), 4),
                "wilcoxon_p_MAE_periodo": pv,
                "motor_gana_total_pct": round(100 * float((at < bt).mean()), 1),
                "wilcoxon_p_error_total": pvt,
            })
    summ_df = pd.DataFrame(summ)
    tests_df = pd.DataFrame(tests)
    summ_df.to_csv(C.RESULTS / "exp3_error_por_largo.csv", index=False)
    tests_df.to_csv(C.RESULTS / "exp3_pruebas_pareadas.csv", index=False)
    print("\nError por largo de historia (WAPE en %):")
    print(summ_df.to_string(index=False))
    print("\nComparaciones pareadas por producto (Wilcoxon de rangos con signo):")
    print(tests_df.to_string(index=False))

    # 3) Censura controlada.
    if len(cdf):
        cdf_ok = cdf[cdf["real_total"].notna()]
        cs = []
        for (H, L, cond), g in cdf_ok.groupby(["H_dias", "L_meses", "condicion"]):
            cs.append({
                "H_dias": H, "L_meses": L, "condicion": cond, "n": len(g),
                "WAPE_total_pct": round(wape(g["ea_total"], g["real_total"]), 2),
                "WAPE_periodos_pct": round(wape(g["ea_periodos"], g["real_total"]), 2),
                "sesgo_total_pct": round(float((g["pron_total"].sum() - g["real_total"].sum()) / g["real_total"].sum() * 100), 2),
                "MdAPE_total_pct": round(float(np.median((g["ea_total"] / g["real_total"])[g["real_total"] > 0]) * 100), 2),
                "pct_sobrepronostico_x3": round(float((g["pron_total"] > 3 * g["real_total"]).mean() * 100), 1),
                "distorsion_historia_pct_media": round(float(g["distorsion_historia_pct"].mean()), 2),
                "dias_censurados_media": round(float(g["dias_censurados"].mean()), 1),
                "unidades_censuradas_total": float(g["unidades_censuradas"].sum()),
                "historia_reparada_menos_observada": float((g["hist_reparada"] - g["hist_observada"]).sum()),
                "periodos_quiebre_media": round(float(g["periodos_quiebre"].mean()), 2),
            })
        cs_df = pd.DataFrame(cs)
        ct = []
        for (H, L), g in cdf_ok.groupby(["H_dias", "L_meses"]):
            p = g.pivot(index="producto", columns="condicion", values="ea_periodos").dropna()
            d = p["C_censurada_marcada"] - p["B_censurada_sin_marcar"]
            try:
                pv = float(wilcoxon(p["C_censurada_marcada"], p["B_censurada_sin_marcar"]).pvalue)
            except ValueError:
                pv = float("nan")
            q = g.pivot(index="producto", columns="condicion", values="distorsion_historia_pct").dropna()
            dq = q["C_censurada_marcada"] - q["B_censurada_sin_marcar"]
            try:
                pq = float(wilcoxon(q["C_censurada_marcada"], q["B_censurada_sin_marcar"]).pvalue)
            except ValueError:
                pq = float("nan")
            ct.append({"H_dias": H, "L_meses": L, "n": len(p),
                       "C_mejor_que_B_pct": round(100 * float((d < 0).mean()), 1),
                       "empate_pct": round(100 * float((d == 0).mean()), 1),
                       "wilcoxon_p": pv,
                       "historia_C_mas_fiel_que_B_pct": round(100 * float((dq < 0).mean()), 1),
                       "wilcoxon_p_historia": pq})
        ct_df = pd.DataFrame(ct)
        cs_df.to_csv(C.RESULTS / "exp3_censura_resumen.csv", index=False)
        ct_df.to_csv(C.RESULTS / "exp3_censura_pruebas.csv", index=False)
        print("\nCENSURA CONTROLADA INYECTADA SOBRE DATOS M5:")
        print(cs_df.to_string(index=False))
        print(ct_df.to_string(index=False))

    lat = df.loc[df["metodo"] == "Motor", "seg_motor"]
    print(f"\nTiempo del motor por producto (s): mediana {lat.median():.3f}, p90 {lat.quantile(0.9):.3f}")
    print(f"Tiempo total: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
