"""Utilidades compartidas de los experimentos de validacion tecnica (OE4.I1).

- Rutas del proyecto y del motor (el motor se importa via PYTHONPATH; NO se modifica).
- Carga de los datos M5 (sales_data1.csv = d_1..d_1000, sales_data2.csv = d_1001..d_1941)
  con cache local en ``cache/``.
- Metricas del paper (MASE / RMSSE escalados por el naive de un paso dentro de muestra,
  Ec. 33-34 de Ye et al., 2024).
- Diagnosticos clasicos de modelos grises (C, p, error relativo medio, grado relacional).
- Registro de entorno (versiones) para los logs.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parents[1]
#: Engine checkout; override with ENGINE_DIR (e.g. a clean checkout of commit be6a98f).
ENGINE = Path(os.environ.get("ENGINE_DIR") or (THESIS / "inventory-dss-ftgm-engine"))
DATA_DIR = ENGINE / "docs" / "ftgmmodel"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"
for _d in (RESULTS, FIGURES, LOGS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

#: Dia d_1 del M5 (calendar.csv de la competencia M5). Se verifica con el cierre de
#: tiendas del 25 de diciembre (ventas ~0) en ``verify_calendar``.
M5_START = date(2011, 1, 29)
N_DAYS = 1941
SEED = 42


def day_dates() -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(M5_START), periods=N_DAYS, freq="D")


# --------------------------------------------------------------------------- datos M5
def load_m5() -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve (meta, ventas) con ventas de forma (30490, 1941), dias d_1..d_1941.

    Une sales_data1.csv (d_1..d_1000) y sales_data2.csv (d_1001..d_1941) por ``id``
    tras verificar que el orden de filas coincide.
    """
    meta_p, sales_p = CACHE / "m5_meta.csv", CACHE / "m5_sales.npy"
    if meta_p.exists() and sales_p.exists():
        return pd.read_csv(meta_p), np.load(sales_p)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d1 = pd.read_csv(DATA_DIR / "sales_data1.csv")
    d2 = pd.read_csv(DATA_DIR / "sales_data2.csv")
    if not (d1["id"].values == d2["id"].values).all():
        d2 = d2.set_index("id").loc[d1["id"]].reset_index()
    assert (d1["id"].values == d2["id"].values).all()
    cols1 = [c for c in d1.columns if c.startswith("d_")]
    cols2 = [c for c in d2.columns if c.startswith("d_")]
    days = cols1 + cols2
    assert days == [f"d_{i}" for i in range(1, N_DAYS + 1)], "columnas de dias inesperadas"
    sales = np.hstack([d1[cols1].to_numpy(np.float64), d2[cols2].to_numpy(np.float64)])
    meta = d1[id_cols].copy()
    meta.to_csv(meta_p, index=False)
    np.save(sales_p, sales)
    return meta, sales


def verify_calendar(sales: np.ndarray) -> dict[str, float]:
    """Verifica d_1 = 2011-01-29: las tiendas Walmart cierran el 25-dic (ventas ~0)."""
    total = sales.sum(axis=0)
    dates = day_dates()
    out: dict[str, float] = {"median_daily_total": float(np.median(total))}
    for y in range(2011, 2016):
        idx = int(np.flatnonzero(dates == pd.Timestamp(date(y, 12, 25)))[0])
        out[f"total_{y}-12-25"] = float(total[idx])
        out[f"total_{y}-12-24"] = float(total[idx - 1])
    return out


# --------------------------------------------------------------- metricas del paper
def paper_scales(insample: np.ndarray) -> tuple[float, float]:
    """Denominadores de MASE y RMSSE: naive de un paso dentro de muestra (Ec. 33-34)."""
    d = np.diff(np.asarray(insample, dtype=float))
    return float(np.mean(np.abs(d))), float(np.mean(d**2))


def grey_diagnostics(actual: np.ndarray, fitted: np.ndarray, rho: float = 0.5) -> dict[str, float]:
    """Pruebas clasicas de precision de modelos grises sobre el ajuste dentro de muestra.

    C   = S2 / S1 (razon de varianza posterior: desv. est. de residuos / desv. est. datos)
    p   = P(|e_k - e_mean| < 0.6745 * S1) (probabilidad de error pequeno)
    RRM = media de |e_k| / x_k (error relativo medio, % ; omite x_k = 0)
    GRD = grado relacional de Deng entre ajuste y real, rho = 0.5:
          xi_k = (Dmin + rho Dmax) / (D_k + rho Dmax), D_k = |x_k - xhat_k|, GRD = media(xi_k)
          (secuencias normalizadas por su valor medio, operador de media, para que el
          grado sea adimensional).
    """
    x = np.asarray(actual, dtype=float)
    f = np.asarray(fitted, dtype=float)
    e = x - f
    s1 = float(np.std(x, ddof=0))
    s2 = float(np.std(e, ddof=0))
    c = s2 / s1 if s1 > 0 else float("nan")
    p = float(np.mean(np.abs(e - e.mean()) < 0.6745 * s1))
    nz = x != 0
    rrm = float(np.mean(np.abs(e[nz]) / np.abs(x[nz])) * 100.0)
    xm, fm = x / np.mean(x), f / np.mean(x)
    delta = np.abs(xm - fm)
    dmin, dmax = float(delta.min()), float(delta.max())
    grd = 1.0 if dmax == 0 else float(np.mean((dmin + rho * dmax) / (delta + rho * dmax)))
    return {"C": c, "p": p, "RRM_pct": rrm, "GRD": grd}


def grey_grade(c: float, p: float) -> str:
    """Grados estandar de precision de modelos grises (prueba de varianza posterior).

    Umbrales y nombres segun la Tabla 2 de Liu et al. (2025), BMC Infectious Diseases 25,
    1020, doi:10.1186/s12879-025-11483-4 (verificado en Crossref y en el texto de PMC12355814):
    I Excelente C<0.35 y P>=0.95; II Bueno C<0.50 y P>=0.80; III Calificado C<0.65 y
    P>=0.70; IV No calificado en otro caso. Se asigna el mejor nivel cuyas dos condiciones
    se cumplen.
    """
    if c < 0.35 and p >= 0.95:
        return "I-Excelente"
    if c < 0.50 and p >= 0.80:
        return "II-Bueno"
    if c < 0.65 and p >= 0.70:
        return "III-Calificado"
    return "IV-No calificado"


# --------------------------------------------------------------------- entorno
def environment() -> str:
    import matplotlib
    import scipy
    import sklearn
    import statsmodels

    try:
        commit = subprocess.run(
            ["git", "-C", str(ENGINE), "log", "-1", "--format=%H %cd"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ENGINE), "status", "--short"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit, dirty = "n/d", ""
    lines = [
        f"python {sys.version.split()[0]} ({platform.platform()})",
        f"cpu_count {os.cpu_count()}",
        f"numpy {np.__version__}, scipy {scipy.__version__}, pandas {pd.__version__}, "
        f"statsmodels {statsmodels.__version__}, scikit-learn {sklearn.__version__}, "
        f"matplotlib {matplotlib.__version__}",
        f"motor inventory-dss-ftgm-engine commit {commit}"
        + (" (con cambios locales sin commit)" if dirty else " (arbol limpio)"),
    ]
    return "\n".join(lines)


class Tee:
    """Duplica stdout a un archivo de log."""

    def __init__(self, path: Path) -> None:
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.f.flush()


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def add_months(d: date, k: int) -> date:
    m = d.month - 1 + k
    return date(d.year + m // 12, m % 12 + 1, 1)


def plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )


#: Paleta sobria (categorica, apta para impresion).
PALETTE = ["#2F5D8A", "#D08C3C", "#5B9A6B", "#A44A4A", "#7A6BA8", "#8C8C8C", "#3E8E9C", "#B5A642", "#C06C84"]
_ = timedelta  # re-export
