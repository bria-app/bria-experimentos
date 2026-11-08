"""H5: variantes de la definicion/agregacion de MASE y RMSSE, calibradas con el SNaive (sin parametros).

Se recalcula el SNaive del Exp. 1 (results/exp1_pronosticos_crudos.csv del Exp. 1) bajo varias
definiciones y se compara con la fila SNaive de la Tabla 1 del paper. La definicion que mejor
reproduce al SNaive se aplica luego a las variantes FTGM para comprobar que la brecha no es de metrica.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

PAPER_SN = {"MASE": {1: (2.12, 1.75), 3: (2.07, 1.71), 6: (2.29, 2.12), 9: (2.72, 2.03), 12: (3.75, 3.25),
                     15: (3.74, 3.02), "1-15": (2.69, 2.35)},
            "RMSSE": {1: (1.81, 1.53), 3: (1.75, 1.54), 6: (1.95, 1.85), 9: (2.26, 1.71), 12: (2.99, 2.50),
                      15: (3.01, 2.54), "1-15": (2.38, 2.04)}}
HS = [1, 3, 6, 9, 12, 15]

Y = pd.read_csv(C.RESULTS / "exp1_series_mensuales.csv", index_col=0).to_numpy().T


def scales(i: int, o: int, kind: str) -> tuple[float, float]:
    y = Y[i]
    if kind == "naive1_insample":
        d = np.diff(y[:o])
    elif kind == "naive1_train37":
        d = np.diff(y[:o - 6])
    elif kind == "naive1_serie63":
        d = np.diff(y)
    elif kind == "snaive12_insample":
        d = y[12:o] - y[:o - 12]
    else:
        raise ValueError(kind)
    return float(np.mean(np.abs(d))), float(np.mean(d ** 2))


def evaluate(raw: pd.DataFrame, scale_kind: str, rmsse_h: str) -> dict:
    """rmsse_h: 'abs' = |e|/sqrt(MSEn) promediado en origenes (Exp. 1);
                'rms' = sqrt(media_origenes(e^2)/MSEn) (RMSE entre origenes a cada h)."""
    out = {}
    recs = []
    for (sid, o), g in raw.groupby(["serie_id", "origen"]):
        smae, smse = scales(sid - 1, o, scale_kind)
        e = (g.sort_values("h")["real"] - g.sort_values("h")["pronostico"]).to_numpy()
        recs.append({"serie_id": sid, "origen": o, "e": e, "smae": smae, "smse": smse})
    df = pd.DataFrame(recs)
    for h in HS:
        mase = df.assign(v=[abs(r.e[h - 1]) / r.smae for r in df.itertuples()]).groupby("serie_id")["v"].mean()
        if rmsse_h == "abs":
            rmsse = df.assign(v=[abs(r.e[h - 1]) / np.sqrt(r.smse) for r in df.itertuples()]).groupby("serie_id")["v"].mean()
        else:
            rmsse = np.sqrt(df.assign(v=[r.e[h - 1] ** 2 / r.smse for r in df.itertuples()]).groupby("serie_id")["v"].mean())
        out[("MASE", h)] = (mase.mean(), mase.median())
        out[("RMSSE", h)] = (rmsse.mean(), rmsse.median())
    mase = df.assign(v=[np.mean(np.abs(r.e)) / r.smae for r in df.itertuples()]).groupby("serie_id")["v"].mean()
    rmsse = df.assign(v=[np.sqrt(np.mean(r.e ** 2) / r.smse) for r in df.itertuples()]).groupby("serie_id")["v"].mean()
    out[("MASE", "1-15")] = (mase.mean(), mase.median())
    out[("RMSSE", "1-15")] = (rmsse.mean(), rmsse.median())
    return out


def main() -> None:
    raw_all = pd.read_csv(C.RESULTS / "exp1_pronosticos_crudos.csv")
    raw = raw_all[raw_all["modelo"] == "SNaive"]
    rows = []
    for sk in ("naive1_insample", "naive1_train37", "naive1_serie63", "snaive12_insample"):
        for rh in ("abs", "rms"):
            res = evaluate(raw, sk, rh)
            err = []
            for metric in ("MASE", "RMSSE"):
                for h in HS + ["1-15"]:
                    pm, pmed = PAPER_SN[metric][h]
                    om, omed = res[(metric, h)]
                    rows.append({"escala": sk, "rmsse_por_h": rh, "metrica": metric, "h": str(h),
                                 "replica_media": round(om, 3), "paper_media": pm,
                                 "replica_mediana": round(omed, 3), "paper_mediana": pmed})
    df = pd.DataFrame(rows)
    df["abs_dif_media"] = (df.replica_media - df.paper_media).abs()
    df["abs_dif_mediana"] = (df.replica_mediana - df.paper_mediana).abs()
    df.to_csv(HERE / "results" / "fid_h5_snaive_metricas.csv", index=False)
    summ = (df[df.h != "12"].groupby(["escala", "rmsse_por_h", "metrica"])[["abs_dif_media", "abs_dif_mediana"]]
            .mean().round(3).reset_index())
    with open(HERE / "logs" / "fid_h5_log.txt", "w", encoding="utf-8") as f:
        txt = ("H5 - SNaive: error absoluto medio vs Tabla 1 (h = 1,3,6,9,15 y 1-15; se excluye h=12 por el "
               "desfase de la tabla del paper)\n" + summ.to_string(index=False))
        sub = df[(df.escala == "naive1_insample")]
        txt += "\n\nDetalle con escala naive1_insample:\n" + sub.to_string(index=False)
        print(txt)
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
