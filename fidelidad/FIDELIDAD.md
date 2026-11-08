# Fidelidad del FTGM frente a la Tabla 1 de Ye et al. (2024): búsqueda de la brecha

Investigación del 02/11/2026 sobre por qué el FTGM del motor no reproduce la exactitud que publica
Ye, Xie, Boylan y Shang (2024, *IJF*, doi:10.1016/j.ijforecast.2023.12.006): MASE 1–15 de 1.65/1.28
(media/mediana) y RMSSE de 1.38/1.04. **El motor no se modificó ni se importó para los pronósticos de
esta carpeta.** Se usó una implementación independiente con "perillas", que con la configuración base
reproduce al motor (diferencia máxima en θ: 0; en el pronóstico: 1.7e-4 del nivel medio, 210 ajustes;
`logs/verifica_log.txt`). Todas las cifras de este documento salen de `results/` y `logs/`.

## Conclusión

1. **Ninguna variante honesta reproduce la Tabla 1.** Se probaron 50 variantes sobre las 70 series
   con el protocolo del Exp. 1. La mejor que es un pronóstico real (*origen_x_diff_refit*, que reinicia
   la EDO en el origen con el último dato observado y **no está en el paper**) obtiene una MASE de
   2.60/2.03 y una RMSSE de 2.36/1.98, es decir, +58 % y +71 % sobre el paper en la media. La variante
   más fiel al código de los autores (*init_matlab_literal*) obtiene 3.99/3.07 en MASE.
2. **Sí se identificó el código que generó los parámetros del paper.** La estimación en dos etapas de
   `ftgm_1.m` (con el valor inicial `eta` literal) sobre los 37 meses de entrenamiento del primer origen
   reproduce la Tabla C.1 en todas las cifras publicadas: θ = [0.005553, 10771.06, 0.001277, −0.002340,
   −826.695, 1135.559] frente a [0.0056, 10771.06, 0.0013, −0.0023, −826.69, 1135.56]. La estimación de
   una etapa del motor se desvía en la segunda cifra decimal (10770.54; −826.81; 1135.78). Esto confirma
   que el paper usó `ftgm_N.m`, con t₁ = 1, h = 1 y T = 12, y que los parámetros del pronóstico se
   estiman sin la validación (lectura literal del Algoritmo 1). Con ese mismo código, integrando la
   Ec. (5) como sistema de dos estados, la MASE es de 3.99/3.07.
3. **La exactitud del paper solo se alcanza con información del periodo de prueba.** Si el modelo se
   estima incluyendo los 15 meses de prueba y se evalúa su *ajuste* en esa ventana (*oraculo_matlab*,
   que es una fuga deliberada), la MASE es de 1.69/1.28 (+2.5 % / 0 %), pero la RMSSE es de 1.51/1.17
   (+9 % / +13 %) y el perfil por horizonte no coincide: en h = 1 da 2.07, frente a 1.09 del paper.
4. **Hay además una inconsistencia en los datos por serie del paper.** La serie 13 (Foods_2_CA_2)
   salta de unas 3 400 a entre 9 035 y 15 768 unidades por mes desde junio de 2015 (mes 53), dentro de
   todas las ventanas de prueba. En la réplica, su RMSSE 1–15 es ≥ 17.05 en los 46 modelos que son
   pronóstico: las 41 variantes FTGM, SNaive, Naive, ETS, SARIMA y el torneo del motor. Ni siquiera los
   oráculos con fuga bajan de 9.17 (`results/fid_huella_por_serie.csv`); solo el ajuste dentro de muestra
   (H10, que nunca ve el salto) la deja benigna (0.54–0.58). Sin embargo, el paper (Sec. 5.1) señala
   como los peores casos del FTGM a las series 40, 39, 20 y 60, con RMSSE de 3.22, 3.10, 3.04 y 2.85.
   Ningún escenario reproduce esa huella: en H10, los peores casos son las series 63, 24 y 36 (MATLAB) o
   36, 55 y 38 (base). Además, la media de la RMSSE del SNaive del paper (2.38)
   coincide con la de la réplica *con* la serie 13 (2.40) y no sin ella (2.19), así que los datos del
   paper también tienen el salto. Si el FTGM del paper tuviera ≈ 17 en esa serie, las otras 69 debieran
   promediar (1.38·70 − 17)/69 ≈ 1.15. Eso está por debajo del oráculo MATLAB sin la serie 13 (1.29).
   Es un indicio fuerte, no una prueba, porque la frase del paper dice "such as".
5. **La métrica no explica la brecha (H5).** La escala del naive de un paso dentro de muestra es la que
   mejor calibra al SNaive del paper (error medio de 0.067 en MASE). Para la RMSSE *por horizonte*, la
   definición del paper es, con mayor probabilidad, la raíz del promedio entre orígenes de e²/MSE_naive.
   Con ella el error del SNaive baja de 0.119 a 0.058 y se explica la diferencia de 0.05–0.27 en la RMSSE
   por horizonte que anota `RESULTADOS.md`. Este ajuste no cambia la MASE ni las cifras 1–15.

**Decisión.** Como ninguna variante legítima queda a menos de ~10 % del paper, **no se propone
parche** (`propuesta_parche.diff` no se creó) **ni se reejecutó el Exp. 3**, conforme al criterio del
encargo. La brecha restante no se puede atribuir a un detalle de implementación del motor. Las causas
compatibles con la evidencia se listan en la última sección.

## Datos y protocolo

Son los del Exp. 1 (`../results/exp1_series_mensuales.csv`): 70 series × 63 meses, orígenes 43–48,
h = 15, MASE y RMSSE escaladas por el naive de un paso dentro de muestra, promedio de 6 orígenes por
serie y luego media/mediana sobre las 70. La agregación es idéntica a `exp1_replicacion_ye.summarize`;
`base_literal` y `base_refit` reproducen al centésimo FTGM_OLS_paper (4.83/3.54) y FTGM_OLS_refit
(3.29/2.55). Si un orden diverge (> 50 × el máximo histórico), se usa el siguiente del ranking; si
divergen todos, el SNaive. Solo hubo sustituciones en la variante `h_1/12_T12`. No hay aleatoriedad
(la semilla 42 se fija por consistencia).

- "literal": parámetros estimados con el tramo de entrenamiento (o − 6) y pronóstico de 6 + 15 pasos,
  de los que se usan los 15 últimos (Algoritmo 1 y Tabla C.1).
- "refit": el orden elegido se re-estima con las o observaciones del in-sample.

## Resultados por hipótesis (MASE y RMSSE, media/mediana; `results/fid_resumen.csv`)

**MASE**

| Variante | h=1 | h=3 | h=6 | h=9 | h=12 | h=15 | 1–15 |
|---|---|---|---|---|---|---|---|
| **Paper (Tabla 1)** | **1.09/0.82** | **1.24/1.00** | **1.30/1.02** | **2.04/1.23** | **2.02/1.28** | **1.99/1.20** | **1.65/1.28** |
| H0 base_literal (= motor, Alg. 1 literal) | 3.57/3.14 | 2.77/2.35 | 3.80/2.47 | 6.39/4.69 | 6.24/4.64 | 5.15/3.20 | 4.83/3.54 |
| H0 base_refit (= motor, re-estimado) | 2.39/2.23 | 2.72/2.46 | 2.98/2.25 | 3.14/2.16 | 3.46/2.30 | 4.77/3.29 | 3.29/2.55 |
| H1 diff_refit (x̂ = ŷ(k) − ŷ(k−1)) | 2.09/1.94 | 2.68/2.26 | 2.92/2.18 | 3.15/2.39 | 3.42/2.40 | 4.68/3.19 | 3.21/2.54 |
| H2/H4 init_matlab_literal (código MATLAB) | 2.81/2.59 | 2.36/2.05 | 2.84/2.27 | 5.29/3.56 | 5.20/3.77 | 4.42/2.93 | 3.99/3.07 |
| H2/H4 init_matlab_refit | 2.25/2.00 | 2.27/2.20 | 2.42/2.19 | 3.10/2.55 | 3.43/2.45 | 4.07/2.98 | 2.99/2.72 |
| H6 matlab_refit_diff | 1.93/1.71 | 2.30/2.13 | 2.38/2.13 | 3.04/2.24 | 3.49/2.57 | 4.06/2.83 | 2.93/2.66 |
| H3 val12_refit (validación de 12) | 2.12/1.93 | 2.76/2.29 | 2.98/2.26 | 3.16/2.18 | 3.43/2.20 | 4.80/3.26 | 3.23/2.57 |
| H3 N1_fijo_refit | 2.01/1.89 | 2.67/2.29 | 3.03/2.46 | 3.09/1.99 | 3.53/2.37 | 4.65/2.94 | 3.20/2.51 |
| H9 origen_x_diff_refit (fuera del paper) | 1.31/1.20 | 1.87/1.66 | 2.14/1.65 | 2.79/1.88 | 3.06/2.27 | 3.84/2.75 | 2.60/2.03 |
| *H8 Ntest_matlab_refit (N elegido con la prueba)* | 1.96/1.64 | 2.07/2.00 | 2.14/1.89 | 2.48/1.73 | 2.84/1.91 | 3.49/2.44 | 2.54/2.16 |
| *H7 oraculo_matlab (estimado con la prueba)* | 2.07/1.42 | 1.87/1.22 | 1.29/1.14 | 1.58/1.24 | 1.76/1.27 | 1.88/1.36 | 1.69/1.28 |
| *H7 oraculo_base* | 1.42/1.12 | 1.19/0.90 | 1.23/1.04 | 1.36/1.02 | 1.41/1.04 | 1.65/1.11 | 1.34/1.02 |
| *H10 ajuste de los últimos 15 meses (MATLAB)* | 2.12/1.83 | 1.92/1.58 | 1.31/1.05 | 1.27/1.00 | 1.28/1.11 | 1.38/1.11 | 1.45/1.20 |

**RMSSE**

| Variante | h=1 | h=3 | h=6 | h=9 | h=12 | h=15 | 1–15 |
|---|---|---|---|---|---|---|---|
| **Paper (Tabla 1)** | **0.92/0.73** | **1.05/0.86** | **1.08/0.82** | **1.65/1.11** | **1.69/0.99** | **1.67/1.06** | **1.38/1.04** |
| H0 base_literal | 2.66/2.53 | 2.06/1.86 | 2.78/1.94 | 4.64/3.76 | 4.54/3.48 | 3.74/2.47 | 4.02/3.29 |
| H0 base_refit | 1.80/1.77 | 2.04/1.90 | 2.21/1.74 | 2.32/1.77 | 2.55/1.79 | 3.52/2.32 | 2.80/2.37 |
| H1 diff_refit | 1.57/1.56 | 2.01/1.77 | 2.17/1.72 | 2.34/1.70 | 2.53/1.79 | 3.45/2.51 | 2.75/2.35 |
| H2/H4 init_matlab_literal | 2.16/1.90 | 1.80/1.57 | 2.17/1.81 | 3.99/2.89 | 3.93/2.95 | 3.34/2.38 | 3.54/3.04 |
| H2/H4 init_matlab_refit | 1.71/1.57 | 1.73/1.64 | 1.84/1.70 | 2.34/2.00 | 2.58/1.88 | 3.09/2.32 | 2.65/2.35 |
| H6 matlab_refit_diff | 1.46/1.41 | 1.76/1.55 | 1.81/1.67 | 2.30/1.71 | 2.62/2.00 | 3.08/2.26 | 2.60/2.34 |
| H3 val12_refit | 1.59/1.52 | 2.06/1.80 | 2.21/1.83 | 2.33/1.69 | 2.53/1.81 | 3.53/2.45 | 2.76/2.35 |
| H3 N1_fijo_refit | 1.50/1.49 | 2.00/1.78 | 2.25/1.88 | 2.28/1.55 | 2.61/1.81 | 3.43/2.28 | 2.72/2.32 |
| H9 origen_x_diff_refit | 1.00/0.91 | 1.42/1.30 | 1.62/1.28 | 2.10/1.44 | 2.30/1.74 | 2.87/2.22 | 2.36/1.98 |
| *H8 Ntest_matlab_refit* | 1.49/1.29 | 1.58/1.49 | 1.64/1.42 | 1.89/1.35 | 2.16/1.42 | 2.67/1.87 | 2.28/1.92 |
| *H7 oraculo_matlab* | 1.58/1.09 | 1.43/0.95 | 0.99/0.88 | 1.21/0.95 | 1.34/0.94 | 1.44/1.01 | 1.51/1.17 |
| *H7 oraculo_base* | 1.09/0.84 | 0.91/0.71 | 0.92/0.83 | 1.02/0.80 | 1.06/0.83 | 1.24/0.90 | 1.20/0.97 |
| *H10 ajuste de los últimos 15 meses (MATLAB)* | 1.59/1.43 | 1.45/1.21 | 0.99/0.84 | 0.96/0.76 | 0.95/0.79 | 1.03/0.83 | 1.27/1.10 |

Las filas en cursiva son diagnósticos con fuga o que no son pronósticos. La tabla 1–15 de las 50
variantes, con la brecha porcentual de cada una, está en `results/fid_tabla_consolidada.csv`. Ninguna
tiene las cuatro cifras 1–15 dentro de ±10 %.

### Detalle por hipótesis

- **H1 (restauración por primeras diferencias de ŷ).** Mejora poco: 3.29 → 3.21 en refit y 4.83 → 4.78
  en la variante literal. Junto con el código MATLAB, 3.99 → 3.95. Descartada como causa.
- **H2 (condición inicial, índice temporal, h, T).**
  - Condición inicial: y(t₁) = x₁ en lugar de b/(1−a) da 3.26. La condición de MATLAB (y(t₁) = x₁,
    x(t₁) = eta, con el desfase constante c = eta − a(t₁)x₁ − b(t₁) que conserva el sistema de dos
    estados) da 2.99 en refit y 3.99 en la variante literal. Es la mejor mejora dentro del paper, pero
    insuficiente.
  - t₀ = 0 o 0.5 da resultados **idénticos** (3.29): desplazar el tiempo solo rota la base de Fourier.
    Solo cambian los θ mostrados, y la Tabla C.1 fija t₁ = 1.
  - Usar el tiempo en años (h = 1/12, T = 1) empeora el resultado (3.79). Escalar mal T (h = 1/12,
    T = 12) lo hace explotar (14.47). Usar ω = 1 da 3.27.
  - El sistema de dos estados integrado directamente con las tolerancias por defecto de ode45 (RelTol
    1e-3) cambia cada pronóstico hasta en un 19 % del nivel (mediana 2.2 %; `verifica_log.txt`), pero el
    agregado casi no se mueve: 3.97 frente a 3.99.
- **H3 (selección del orden).**
  - Validación de 3, 12 o 15 meses: 3.30, 3.23 y 3.25.
  - N ≤ 3 o N ≤ 6: 3.27 y 3.29.
  - Criterio MAE o MAPE: 3.27; error de ajuste en lugar de validación: 3.42.
  - N fijo en 1 o 2: 3.20 y 3.31.
  - Validación de 12 con el código MATLAB literal: 6.30.

  Ninguna opción se acerca. Elegir N **con la propia ventana de prueba** (fuga parcial) tampoco basta:
  2.54 con MATLAB re-estimado.
- **H4 (forma de la estimación).** Quitar x₁ de z da 3.35. La estimación MATLAB en dos etapas con la
  condición inicial del paper da 3.28; con eta evaluado correctamente en t₁, 3.02. El orden seno/coseno
  y que a(t) y b(t) compartan N no tienen efecto: son reparametrizaciones de la misma regresión lineal.
- **H5 (métrica).** Se resume en el punto 5 de la conclusión (`results/fid_h5_snaive_metricas.csv`).
  La escala estacional (tipo M4) queda descartada: error de 1.17 en la MASE del SNaive.

## Causas restantes compatibles con la evidencia

1. **El paper no publica `odefcn`** (las funciones que llama `ftgm_N.m` no están en el repositorio) ni
   el script de evaluación por orígenes, y `ftgm.mat` está vacío (2 bytes, solo un salto de línea).
   Por eso no hay pronósticos de los autores para comparar numéricamente. Se supuso que `odefcn` es la
   Ec. (5) en forma de dos estados [y, x], que es lo coherente con `X0 = [xx(1) eta]`. Un `odefcn`
   distinto de la Ec. (5) es la única diferencia de implementación que quedó sin probar.
2. **Fuga de información del periodo de prueba en el pipeline de evaluación del paper.** Las cifras
   1–15 del paper están en el rango del ajuste dentro de muestra (1.45/1.20) y del oráculo (1.34–1.69),
   y no en el de cualquier pronóstico honesto (≥ 2.54 incluso eligiendo N con la prueba). La huella de
   la serie 13 tampoco cuadra con un pronóstico. Pero el perfil por horizonte del oráculo no coincide
   con el del paper, así que el mecanismo exacto no se pudo identificar.
3. **Error de transcripción o de agregación en la Tabla 1** (como el desfase observado en la columna
   h = 12 del SNaive). No explicaría por sí solo una brecha del doble.

Para citar la Tabla 1 en la tesis se recomienda declarar que **la exactitud del FTGM del paper no es
reproducible con el código publicado por los autores**, aunque los parámetros de la Tabla C.1 sí se
reproducen exactamente con ese código.

## Ideas para el motor (no son parches y no reproducen el paper)

Dos variantes honestas mejoran al motor en esta muestra, pero ninguna supera a ETS (2.43) ni a SARIMA
(2.28) del Exp. 1:

- Estimación en dos etapas con la condición inicial de MATLAB: refit de 3.29 a 2.99.
- Reinicio de la EDO en el origen con x(t_m) observado: 2.60–2.68.

Si se quisieran incorporar, habría que evaluarlas como mejoras propias y no como fidelidad al paper.

## Reproducción

Desde `_tesis_v2/experimentos/fidelidad`, con Python 3.11.4 (mismo entorno que el Exp. 1). Se requiere
`../results/exp1_series_mensuales.csv`. La corrida completa tarda 5 min 2 s con 16 procesos.

```bash
python verifica.py        # BASE == motor; Tabla C.1 por variante; EDO de 2 estados con tolerancia de ode45
python fid_run.py         # 50 variantes en 11 grupos (H0-H10); se puede pasar un subconjunto: python fid_run.py H2_inicial_tiempo
python h5_metricas.py     # H5: definiciones de métrica calibradas con el SNaive
python consolidar.py      # tabla consolidada + huella por serie
```

| Archivo | Contenido |
|---|---|
| `ftgm_variantes.py` | FTGM independiente con perillas (estimación paper/MATLAB/sin x₁, condición inicial, restauración, t₀, h, T, ridge, sistema de 2 estados, reinicio en el origen) y Algoritmo 1 parametrizable |
| `fid_run.py` | grupos de variantes, protocolo de orígenes, diagnósticos de fuga (H7, H8, H10) y resumen |
| `verifica.py` | equivalencia con el motor, Tabla C.1 y verificación de la EDO de 2 estados |
| `h5_metricas.py` | definiciones alternativas de MASE y RMSSE sobre el SNaive del Exp. 1 |
| `consolidar.py` | `results/fid_tabla_consolidada.csv` y `results/fid_huella_por_serie.csv` |
| `results/fid_crudos_<grupo>.csv` | pronósticos crudos por serie, origen, variante y horizonte |
| `results/fid_ordenes_<grupo>.csv` | orden N usado y sustituciones |
| `results/fid_metricas_por_serie_<grupo>.csv` | MASE y RMSSE por serie |
| `results/fid_resumen.csv`, `results/fid_tabla_consolidada.csv` | resúmenes |
| `results/fid_tabla_c1.csv` | θ por variante de estimación (37, 42 y 43 observaciones) |
| `logs/*.txt` | salida literal de cada script (incluye el entorno y el commit del motor) |
