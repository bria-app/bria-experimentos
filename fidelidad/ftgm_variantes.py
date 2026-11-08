"""Variantes del FTGM para investigar la brecha con la Tabla 1 de Ye et al. (2024).

Implementacion INDEPENDIENTE del motor (el motor no se modifica ni se importa aqui): un FTGM
parametrizado por "perillas" que cubren las hipotesis H1-H4 del encargo y la lectura del
codigo MATLAB de los autores (docs/ftgmmodel/utils/ftgm_N.m). Con las perillas por defecto
(`BASE`) reproduce el motor (se verifica contra app.ftgm en `verifica_motor`).

Perillas (dict `cfg`):
  est      'paper'  : MCO de una etapa sobre z_k = (y_{k-1}+y_k)/2 + (h/2) x_1   (Ec. 11-15, motor)
           'matlab' : dos etapas de ftgm_N.m (etapa 1 sin x_1 en z y con productos trig. promediados;
                      eta = (pa1+sum(alfa))x_1 + pa2+sum(beta); etapa 2 con z + (r/2) eta)
           'nox1'   : una etapa, z sin el termino (h/2) x_1
  init     'paper'  : y(t1) = b(t1)/(1-a(t1))                     (Ec. A.6, motor)
           'x1'     : y(t1) = x_1 (eta_y observado), x(t1)=a y+b  (consistente)
           'matlab' : y(t1) = x_1, x(t1) = eta (sistema de 2 estados de ftgm_N.m: x = a y + b + c)
  restore  'inst'   : x^(t) = a(t) y(t) + b(t)                    (motor)
           'diff'   : x^(t_k) = y(t_k) - y(t_{k-1})               (H1, restauracion IAGO)
  t0       instante del primer dato (1 = motor; 0; 0.5 ...)       (H2)
  h        intervalo de muestreo en unidades de tiempo (1 = motor) (H2)
  T        periodo en unidades de tiempo (12 = motor)              (H2)
  ridge    0 = MCO
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(est="paper", init="paper", restore="inst", t0=1.0, h=1.0, T=12.0, ridge=0.0,
            rtol=1e-8, atol=1e-6)
GROWTH = 50.0  # misma salvaguarda de divergencia del motor (max_growth_factor)


class Diverge(Exception):
    pass


def harm(t: np.ndarray, N: int, w: float) -> np.ndarray:
    n = np.arange(1, N + 1)
    ang = w * np.outer(t, n)
    H = np.empty((t.size, 2 * N))
    H[:, 0::2] = np.cos(ang)
    H[:, 1::2] = np.sin(ang)
    return H


def ab(t: np.ndarray, th: np.ndarray, N: int, w: float) -> tuple[np.ndarray, np.ndarray]:
    H = harm(np.atleast_1d(t), N, w)
    return th[0] + H @ th[2:2 + 2 * N], th[1] + H @ th[2 + 2 * N:2 + 4 * N]


def _ls(D: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        return np.linalg.lstsq(D, target, rcond=None)[0]
    sc = np.linalg.norm(D, axis=0)
    sc[sc == 0] = 1.0
    wts = np.full(D.shape[1], np.sqrt(ridge))
    wts[:2] = 0.0
    th = np.linalg.lstsq(np.vstack([D / sc, np.diag(wts)]), np.concatenate([target, np.zeros(D.shape[1])]),
                         rcond=None)[0]
    return th / sc


def estimate(x: np.ndarray, N: int, cfg: dict) -> tuple[np.ndarray, float | None]:
    """Devuelve (theta, eta). eta solo para est='matlab'."""
    m = x.size
    h, w = cfg["h"], 2 * np.pi / cfg["T"]
    t = cfg["t0"] + h * np.arange(m)
    # Y1 = cumsum([1; diff(ts)] .* xx)  (ftgm_N.m): y(t1) = x1, luego pasos h
    Y = np.cumsum(np.concatenate([[1.0], np.full(m - 1, h)]) * x)
    bg = 0.5 * (Y[:-1] + Y[1:])
    F = harm(t[1:], N, w)
    if cfg["est"] in ("paper", "nox1"):
        z = bg + (0.5 * h * x[0] if cfg["est"] == "paper" else 0.0)
        D = np.hstack([z[:, None], np.ones((m - 1, 1)), F * z[:, None], F])
        return _ls(D, x[1:], cfg["ridge"]), None
    if cfg["est"] == "matlab":
        Fm = harm(t, N, w)
        pa_y = 0.5 * (Y[:-1, None] * Fm[:-1] + Y[1:, None] * Fm[1:])
        pa_f = 0.5 * (Fm[:-1] + Fm[1:])
        A = np.hstack([bg[:, None], np.ones((m - 1, 1)), pa_y, pa_f])
        pa = np.linalg.lstsq(A, x[1:], rcond=None)[0]
        if cfg.get("eta_mode", "literal") == "literal":  # suma de TODOS los coeficientes (codigo MATLAB)
            eta = (pa[0] + pa[2:2 + 2 * N].sum()) * x[0] + pa[1] + pa[2 + 2 * N:].sum()
        else:  # a(t1) x1 + b(t1) evaluado correctamente
            a1, b1 = ab(t[:1], pa, N, w)
            eta = float(a1[0] * x[0] + b1[0])
        z = bg + 0.5 * h * eta
        D = np.hstack([z[:, None], np.ones((m - 1, 1)), F * z[:, None], F])
        return _ls(D, x[1:], cfg["ridge"]), float(eta)
    raise ValueError(cfg["est"])


def trajectory(x: np.ndarray, th: np.ndarray, eta: float | None, N: int, cfg: dict, n_total: int) -> np.ndarray:
    """Ajuste + pronostico x^ en t_1..t_{n_total} integrando desde t_1."""
    h, w = cfg["h"], 2 * np.pi / cfg["T"]
    t = cfg["t0"] + h * np.arange(n_total)
    a1, b1 = ab(t[:1], th, N, w)
    c = 0.0
    if cfg["init"] == "paper":
        den = 1.0 - a1[0]
        y0 = b1[0] / (den if abs(den) > 1e-8 else 1e-8)
    elif cfg["init"] == "x1":
        y0 = x[0]
    elif cfg["init"] == "matlab":
        y0 = x[0]
        e = eta if eta is not None else a1[0] * x[0] + b1[0]
        c = e - a1[0] * y0 - b1[0]  # invariante del sistema de 2 estados: x - a y - b = c
    else:
        raise ValueError(cfg["init"])
    a0, b0 = th[0], th[1] + c
    ac, bc = th[2:2 + 2 * N], th[2 + 2 * N:2 + 4 * N]
    fr = np.arange(1, N + 1) * w
    if cfg.get("two_state"):
        # Sistema de 2 estados de la Ec. (5) integrado tal cual (X = [y, x]), como ode45 en ftgm_N.m.
        def f2(tt, X):
            cs, sn = np.cos(fr * tt), np.sin(fr * tt)
            hh = np.empty(2 * N); hh[0::2] = cs; hh[1::2] = sn
            dh = np.empty(2 * N); dh[0::2] = -fr * sn; dh[1::2] = fr * cs
            return [X[1], (a0 + ac @ hh) * X[1] + (ac @ dh) * X[0] + bc @ dh]
        x0 = (a1[0] * y0 + b1[0] + c)
        sol = solve_ivp(f2, (t[0], t[-1]), [y0, x0], t_eval=t, method="RK45", rtol=cfg["rtol"], atol=cfg["atol"])
        if not sol.success:
            raise Diverge(sol.message)
        if cfg["restore"] == "diff":
            xh = np.empty(n_total); xh[0] = x[0]; xh[1:] = np.diff(sol.y[0]) / h
            return xh
        return sol.y[1]

    def rhs(tt, y):
        hh = np.empty(2 * N)
        hh[0::2] = np.cos(fr * tt)
        hh[1::2] = np.sin(fr * tt)
        return [(a0 + ac @ hh) * y[0] + b0 + bc @ hh]

    if cfg["restore"] == "diff":
        tg = np.concatenate([[t[0] - h], t])  # y(t_0) para la primera diferencia
        sol = solve_ivp(rhs, (t[0], tg[-1]), [y0], t_eval=t, method="RK45", rtol=cfg["rtol"], atol=cfg["atol"])
        if not sol.success:
            raise Diverge(sol.message)
        yy = sol.y[0]
        xh = np.empty(n_total)
        xh[0] = x[0]
        xh[1:] = (yy[1:] - yy[:-1]) / h
        return xh
    sol = solve_ivp(rhs, (t[0], t[-1]), [y0], t_eval=t, method="RK45", rtol=cfg["rtol"], atol=cfg["atol"])
    if not sol.success:
        raise Diverge(sol.message)
    a, b = ab(t, th, N, w)
    return a * sol.y[0] + b + c


def restart_at_origin(x: np.ndarray, th: np.ndarray, N: int, cfg: dict, n_out: int) -> np.ndarray:
    """DIAGNOSTICO (no es del paper): reinicia el sistema de 2 estados (y, x) en el origen t_m con
    y(t_m) = x_1 + integral trapezoidal observada y x(t_m) = x_m observado (invariante c = x_m - a y_m - b),
    y lo integra hacia adelante n_out pasos."""
    h, w = cfg["h"], 2 * np.pi / cfg["T"]
    m = x.size
    tm = cfg["t0"] + h * (m - 1)
    tg = tm + h * np.arange(n_out + 1)
    y_m = x[0] + h * float(np.sum(0.5 * (x[:-1] + x[1:])))
    am, bm = ab(np.array([tm]), th, N, w)
    c = (x[-1] - am[0] * y_m - bm[0]) if cfg["init"] == "origin_x" else 0.0
    fr = np.arange(1, N + 1) * w
    ac, bc = th[2:2 + 2 * N], th[2 + 2 * N:2 + 4 * N]

    def rhs(tt, y):
        hh = np.empty(2 * N)
        hh[0::2] = np.cos(fr * tt)
        hh[1::2] = np.sin(fr * tt)
        return [(th[0] + ac @ hh) * y[0] + th[1] + c + bc @ hh]

    sol = solve_ivp(rhs, (tg[0], tg[-1]), [y_m], t_eval=tg, method="RK45", rtol=cfg["rtol"], atol=cfg["atol"])
    if not sol.success:
        raise Diverge(sol.message)
    if cfg["restore"] == "diff":
        return np.diff(sol.y[0]) / h
    a, b = ab(tg, th, N, w)
    return (a * sol.y[0] + b + c)[1:]


def fit_forecast(x: np.ndarray, N: int, n_out: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    if x.size - 1 < 2 + 4 * N:
        raise Diverge("pocos datos")
    th, eta = estimate(x, N, cfg)
    if not np.all(np.isfinite(th)):
        raise Diverge("theta no finito")
    if cfg["init"] in ("origin_x", "origin_y"):
        fit = trajectory(x, th, eta, N, {**cfg, "init": "matlab" if cfg["est"] == "matlab" else "paper"}, x.size)
        xh = np.concatenate([fit, restart_at_origin(x, th, N, cfg, n_out)])
    else:
        xh = trajectory(x, th, eta, N, cfg, x.size + n_out)
    ref = float(np.max(np.abs(x))) or 1.0
    if not np.all(np.isfinite(xh)) or np.max(np.abs(xh)) > GROWTH * ref:
        raise Diverge("diverge")
    return xh[:x.size], np.clip(xh[x.size:], 0.0, None)


# ------------------------------------------------------------------ seleccion de orden
def _crit(act: np.ndarray, fc: np.ndarray, kind: str) -> float:
    e = act - fc
    if kind == "rmse":
        return float(np.sqrt(np.mean(e ** 2)))
    if kind == "mae":
        return float(np.mean(np.abs(e)))
    if kind == "mape":
        return float(np.mean(np.abs(e) / np.abs(act)))
    raise ValueError(kind)


def select_and_forecast(ins: np.ndarray, H: int, cfg: dict, sel: dict) -> tuple[np.ndarray, dict]:
    """Algoritmo 1 con perillas de seleccion (H3).

    sel: val (l), nmax, crit ('rmse','mae','mape','fit_rmse' = error de ajuste in-sample),
         final ('train' = parametros del tramo de entrenamiento, literal; 'refit' = todo el in-sample),
         fixed_N (si se da, sin seleccion).
    """
    val, nmax, crit, final = sel.get("val", 6), sel.get("nmax", 5), sel.get("crit", "rmse"), sel.get("final", "train")
    if sel.get("fixed_N"):
        ranking = [sel["fixed_N"]]
        scores = {}
    elif sel.get("only_N"):  # parametros del tramo de entrenamiento con N dado (sin seleccion)
        try:
            _, fc = fit_forecast(ins[:-val], sel["only_N"], val + H, cfg)
            return fc[val:], {"N": sel["only_N"], "fallo": ""}
        except Diverge:
            return np.full(H, np.nan), {"N": 0, "fallo": "diverge"}
    else:
        train, valid = ins[:-val], ins[-val:]
        scores = {}
        for N in range(1, nmax + 1):
            try:
                fit, fc = fit_forecast(train, N, val, cfg)
                scores[N] = _crit(train, fit, "rmse") if crit == "fit_rmse" else _crit(valid, fc, crit)
            except Diverge:
                scores[N] = np.inf
        ranking = sorted(scores, key=lambda k: scores[k])
    for N in ranking:
        if not np.isfinite(scores.get(N, 0.0)):
            break
        try:
            if final == "train" and not sel.get("fixed_N"):
                _, fc = fit_forecast(ins[:-val], N, val + H, cfg)
                fc = fc[val:]
            else:
                _, fc = fit_forecast(ins, N, H, cfg)
            return fc, {"N": N, "fallo": ""}
        except Diverge as exc:
            last = str(exc)
            continue
    n = ins.size
    return np.array([ins[n - 12 + ((k - 1) % 12)] for k in range(1, H + 1)]), {"N": 0, "fallo": "snaive"}
