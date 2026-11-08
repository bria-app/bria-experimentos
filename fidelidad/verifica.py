"""Verificaciones previas: (1) la implementacion BASE reproduce al motor; (2) que variante de
estimacion reproduce mejor la Tabla C.1 del paper (Hobbies_1_CA_1, N = 1, primer origen)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import ftgm_variantes as V  # noqa: E402
from fid_run import load_series  # noqa: E402

PAPER_C1 = np.array([0.0056, 10771.06, 0.0013, -0.0023, -826.69, 1135.56])
out = open(HERE / "logs" / "verifica_log.txt", "w", encoding="utf-8")


def p(s=""):
    print(s)
    out.write(s + "\n")


Y, names = load_series()
p(C.environment())
# (1) motor vs BASE
from app.ftgm import FTGM, FTGMConfig, FTGMError  # noqa: E402

difs = []
for i in range(70):
    for N in (1, 2, 3):
        x = Y[i][:37]
        try:
            m = FTGM(order=N, config=FTGMConfig(period=12)).fit(x)
            fc = m.predict(21).point
        except FTGMError:
            continue
        th, _ = V.estimate(x, N, V.BASE)
        fit, fc2 = V.fit_forecast(x, N, 21, V.BASE)
        difs.append({"serie": i + 1, "N": N, "dif_theta_rel": float(np.max(np.abs(th - m.theta_) / (np.abs(m.theta_) + 1e-12))),
                     "dif_fc_rel_nivel": float(np.max(np.abs(fc - fc2)) / np.mean(x))})
d = pd.DataFrame(difs)
p(f"(1) BASE vs motor en {len(d)} ajustes: max dif. relativa theta {d.dif_theta_rel.max():.2e}; "
  f"max dif. pronostico / nivel {d.dif_fc_rel_nivel.max():.2e}")

# (2) Tabla C.1
rows = []
x0 = Y[0]
for n in (37, 42, 43):
    for lab, cfg in [("paper/motor (z con x1)", V.BASE), ("sin x1 en z", {**V.BASE, "est": "nox1"}),
                     ("MATLAB 2 etapas (eta literal)", {**V.BASE, "est": "matlab"}),
                     ("MATLAB 2 etapas (eta evaluado)", {**V.BASE, "est": "matlab", "eta_mode": "eval"}),
                     ("t0 = 0", {**V.BASE, "t0": 0.0})]:
        th, eta = V.estimate(x0[:n], 1, cfg)
        err = np.abs(np.round(th, 4 if True else 2) - PAPER_C1)
        rel = float(np.max(np.abs(th - PAPER_C1) / np.abs(PAPER_C1)))
        rows.append({"n_obs": n, "variante": lab, **{f"th{j}": th[j] for j in range(6)},
                     "eta": eta, "max_dif_rel_vs_paper": rel})
c1 = pd.DataFrame(rows)
c1.to_csv(HERE / "results" / "fid_tabla_c1.csv", index=False)
pd.set_option("display.width", 200)
p("(2) Tabla C.1 (paper: " + ", ".join(f"{v:g}" for v in PAPER_C1) + ")")
p(c1.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

# (3) Sistema de 2 estados de la Ec. (5) integrado directamente (como ode45 con X0 = [x1, eta] y
# tolerancias por defecto de MATLAB: RelTol 1e-3, AbsTol 1e-6) frente a la reduccion por el
# invariante x - a y - b = c que usa ftgm_variantes (init='matlab').
from scipy.integrate import solve_ivp  # noqa: E402

cfg = {**V.BASE, "est": "matlab", "init": "matlab"}
w = 2 * np.pi / 12
mx = []
for i in range(70):
    x = Y[i][:37]
    for N in range(1, 6):
        try:
            _, fc_red = V.fit_forecast(x, N, 21, cfg)
        except V.Diverge:
            continue
        th, eta = V.estimate(x, N, cfg)
        ac, bc = th[2:2 + 2 * N], th[2 + 2 * N:]
        fr = np.arange(1, N + 1) * w

        def f(t, X, th=th, ac=ac, bc=bc, fr=fr):
            c, s = np.cos(fr * t), np.sin(fr * t)
            hh = np.empty(2 * len(fr)); hh[0::2] = c; hh[1::2] = s
            dh = np.empty(2 * len(fr)); dh[0::2] = -fr * s; dh[1::2] = fr * c
            a = th[0] + ac @ hh
            da, db = ac @ dh, bc @ dh
            return [X[1], a * X[1] + da * X[0] + db]

        tt = np.arange(1, 59, dtype=float)
        sol = solve_ivp(f, (1, 58), [x[0], eta], t_eval=tt, method="RK45", rtol=1e-3, atol=1e-6)
        fc2 = np.clip(sol.y[1][37:], 0, None)
        mx.append(float(np.max(np.abs(fc2 - fc_red)) / np.mean(x)))
p(f"(3) EDO de 2 estados (rtol 1e-3, por defecto de ode45) vs reduccion por invariante: {len(mx)} ajustes; "
  f"max dif. pronostico / nivel = {max(mx):.2e}; mediana {np.median(mx):.2e}")
out.close()
