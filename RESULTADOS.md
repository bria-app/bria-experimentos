# Resultados de la validación técnica (OE4.I1)

Benchmark ejecutado el 02/11/2026 sobre el motor `inventory-dss-ftgm-engine` (commit
`be6a98f8bb3a6ebc2c00f46a0ef7a0dc0c28e456`, árbol limpio; el motor no se modificó). **Todas las cifras de
este documento se copian de `results/*.csv` y `logs/*.txt`, generados por los scripts de esta carpeta.**
Ninguna cifra es estimada ni ilustrativa.

**Entorno:** Python 3.11.4 (Windows 11, 16 núcleos lógicos); numpy 2.4.4, scipy 1.16.3, pandas 2.3.3,
statsmodels 0.14.6, scikit-learn 1.7.2, matplotlib 3.10.7 (cabecera de cada `logs/exp*_log.txt`).

**Referencia replicada:** Ye, L., Xie, N., Boylan, J. E., & Shang, Z. (2024). Forecasting seasonal demand
for retail: A Fourier time-varying grey model. *International Journal of Forecasting*.
https://doi.org/10.1016/j.ijforecast.2023.12.006

## 0. Hallazgos sobre los datos (verificados por código)

- `sales_data1.csv` y `sales_data2.csv` **no** son copias: el primero contiene los días d_1–d_1000 y el
  segundo d_1001–d_1941, con las mismas 30 490 filas en el mismo orden de `id`. `common.load_m5()` los une
  (los `id` terminan en `_evaluation`, versión de evaluación del M5 de 1941 días).
- d_1 = 2011-01-29 se verificó con el cierre de tiendas del 25 de diciembre: la venta total de los
  25-dic de 2011 a 2015 fue 13, 11, 20, 20 y 14 unidades, frente a una mediana diaria de 33 830 unidades
  y 29 958–36 227 unidades los 24-dic (`logs/verificaciones_log.txt`, script `verificaciones.py`).

---

## Experimento 1 — Réplica de fidelidad del FTGM (RQ1)

### Objetivo
Comprobar si la implementación del FTGM del motor reproduce la exactitud que reporta la Tabla 1 de Ye et al.
(2024) con el mismo conjunto de datos y protocolo.

### Datos
70 series departamento × tienda del M5 (7 departamentos × 10 tiendas, orden del Apéndice B del paper),
agregadas a **meses calendario completos**: feb-2011 a abr-2016 = **63 observaciones** (se descartan
ene-2011, con 3 días, y may-2016, con 22 días). Archivo: `results/exp1_series_mensuales.csv`.

**Verificación de la construcción mensual:**
1. *Tabla C.1 del paper* (parámetros del FTGM N = 1, Hobbies_1_CA_1, primer origen). Estimando con las
   primeras 37 observaciones (43 del in-sample menos 6 de validación), el motor obtiene
   θ = [0.005553, 10770.54, 0.001277, −0.002340, −826.81, 1135.78] frente a
   θ = [0.0056, 10771.06, 0.0013, −0.0023, −826.69, 1135.56] del paper (`results/exp1_tabla_c1.csv`).
   Coinciden en 3–4 cifras significativas; con 42 o 43 observaciones la coincidencia es menor. Esto
   confirma la construcción mensual y que, en el paper, los parámetros del primer origen se estiman solo con
   el tramo de entrenamiento (Algoritmo 1, línea 9).
2. *Naive estacional* (modelo sin parámetros): MASE 1–15 de la réplica 2.74/2.40 (media/mediana) frente a
   2.69/2.35 del paper; RMSSE 1–15 2.40/2.07 frente a 2.38/2.04 (`results/exp1_brecha_vs_paper.csv`).

### Protocolo (Sección 4.4 del paper)
- Rolling origin expansivo: orígenes 43, 44, …, 48; horizonte 15; 6 × 15 pronósticos por serie.
- MASE y RMSSE escalados por el naive de **un paso** dentro de muestra (Ec. 33–34): para cada horizonte h
  se calcula |e_h|/MAE_naive y |e_h|/√MSE_naive; para "1–15" se calcula la métrica sobre los 15 pasos de
  cada origen. Se promedian los 6 orígenes por serie y luego se toma media y mediana sobre las 70 series.
  La variante con media/mediana sobre los 420 valores (70 × 6) está en
  `results/exp1_tabla_resumen_agrupada420.csv` y no cambia las conclusiones.
- Modelos (código en `exp1_replicacion_ye.py`):

| Modelo | Definición |
|---|---|
| FTGM_OLS_paper | Algoritmo 1 del paper: validación = últimas 6 obs del in-sample, N ∈ {1,…,5}, MCO (λ = 0), T = 12; parámetros estimados con el tramo de entrenamiento (o − 6 obs) y pronóstico integrando la EDO hasta t = o + 15. Lectura literal del Algoritmo 1, respaldada por la Tabla C.1. |
| FTGM_OLS_refit | Igual, pero el orden elegido se re-estima con todo el in-sample (o obs). |
| FTGM_Ridge | Algoritmo 1 del motor con rejilla ridge λ ∈ {0; 0.5; 3; 10}, re-estimado con o obs. |
| FTGM_OLS_ancla | **Diagnóstico, no es del paper:** como FTGM_OLS_refit, pero la EDO continúa desde la suma acumulada observada en el origen. |
| Motor_torneo | `ForecastService` de producción (modo legado `period=12`, `horizon=15`) con la competencia de modelos. |
| SNaive | ŷ(n+h) = y(n+h−12·⌈h/12⌉). |
| Naive | Último valor. |
| ETS | `statsmodels` ETSModel; mínimo AICc entre 9 especificaciones: (A,N,N), (A,A,N), (A,Ad,N), (A,N,A), (A,A,A), (A,Ad,A), (M,N,M), (M,A,M), (M,Ad,M). Comparable con HWES del paper. |
| SARIMA | `statsmodels` SARIMAX; D = 1; d ∈ {0, 1} por KPSS (α = 0.05) sobre la serie con diferencia estacional; p, q ∈ {0,1,2}, P, Q ∈ {0,1}; constante si d = 0; mínimo AICc. |

Salvaguarda de ETS y SARIMA: se descarta una especificación cuyo pronóstico no sea finito o supere 5 veces
el máximo histórico. En una primera corrida sin salvaguarda, un ajuste SARIMA de un origen de la serie 35
explotó (MASE medio del SARIMA 1216.66); con la salvaguarda se descartó una especificación en 1 de 420
orígenes y ninguna en ETS (`logs/exp1_log.txt`).

### Resultados (`results/exp1_tabla_resumen.csv`, `logs/exp1_log.txt`)

**MASE (media/mediana sobre 70 series)**

| Modelo | h=1 | h=3 | h=6 | h=9 | h=12 | h=15 | h=1–15 |
|---|---|---|---|---|---|---|---|
| **FTGM (paper, Tabla 1)** | **1.09/0.82** | **1.24/1.00** | **1.30/1.02** | **2.04/1.23** | **2.02/1.28** | **1.99/1.20** | **1.65/1.28** |
| FTGM_OLS_paper | 3.57/3.14 | 2.77/2.35 | 3.80/2.47 | 6.39/4.69 | 6.24/4.64 | 5.15/3.20 | 4.83/3.54 |
| FTGM_OLS_refit | 2.39/2.23 | 2.72/2.46 | 2.98/2.25 | 3.14/2.16 | 3.46/2.30 | 4.77/3.29 | 3.29/2.55 |
| FTGM_Ridge | 2.21/2.09 | 2.54/2.18 | 2.71/2.02 | 3.10/2.33 | 3.66/2.36 | 4.38/3.18 | 3.15/2.51 |
| FTGM_OLS_ancla | 2.31/2.22 | 2.62/2.43 | 2.85/2.28 | 3.01/2.18 | 3.32/2.26 | 4.60/3.27 | 3.16/2.54 |
| Motor_torneo | 1.46/1.18 | 1.78/1.58 | 2.20/1.93 | 2.94/2.36 | 3.20/2.41 | 3.85/2.98 | 2.65/2.26 |
| SNaive | 2.15/1.77 | 2.16/1.77 | 2.36/2.15 | 2.81/2.11 | 2.84/2.01 | 3.81/2.90 | 2.74/2.40 |
| Naive | 1.31/1.20 | 1.90/1.65 | 2.29/1.94 | 3.05/2.54 | 2.84/2.01 | 3.64/2.83 | 2.57/2.30 |
| ETS | 1.24/1.15 | 1.71/1.59 | 2.12/1.77 | 2.74/2.10 | 2.88/2.21 | 3.52/2.67 | 2.43/2.05 |
| SARIMA | 1.29/1.16 | 1.71/1.50 | 1.87/1.42 | 2.42/1.69 | 2.65/1.81 | 3.36/2.36 | 2.28/1.68 |

**RMSSE (media/mediana sobre 70 series)**

| Modelo | h=1 | h=3 | h=6 | h=9 | h=12 | h=15 | h=1–15 |
|---|---|---|---|---|---|---|---|
| **FTGM (paper, Tabla 1)** | **0.92/0.73** | **1.05/0.86** | **1.08/0.82** | **1.65/1.11** | **1.69/0.99** | **1.67/1.06** | **1.38/1.04** |
| FTGM_OLS_paper | 2.66/2.53 | 2.06/1.86 | 2.78/1.94 | 4.64/3.76 | 4.54/3.48 | 3.74/2.47 | 4.02/3.29 |
| FTGM_OLS_refit | 1.80/1.77 | 2.04/1.90 | 2.21/1.74 | 2.32/1.77 | 2.55/1.79 | 3.52/2.32 | 2.80/2.37 |
| FTGM_Ridge | 1.67/1.73 | 1.91/1.72 | 2.01/1.59 | 2.29/1.75 | 2.70/1.92 | 3.24/2.52 | 2.64/2.15 |
| FTGM_OLS_ancla | 1.75/1.70 | 1.99/1.88 | 2.14/1.73 | 2.26/1.75 | 2.48/1.75 | 3.43/2.32 | 2.74/2.36 |
| Motor_torneo | 1.11/0.91 | 1.37/1.22 | 1.69/1.55 | 2.26/1.82 | 2.44/1.94 | 2.93/2.42 | 2.34/2.03 |
| SNaive | 1.62/1.33 | 1.64/1.39 | 1.80/1.69 | 2.15/1.66 | 2.17/1.58 | 2.87/2.27 | 2.40/2.07 |
| Naive | 1.00/0.92 | 1.45/1.29 | 1.76/1.59 | 2.33/2.07 | 2.17/1.58 | 2.76/2.28 | 2.30/2.10 |
| ETS | 0.95/0.88 | 1.31/1.22 | 1.64/1.41 | 2.11/1.65 | 2.20/1.68 | 2.67/2.12 | 2.18/1.85 |
| SARIMA | 0.98/0.87 | 1.31/1.14 | 1.43/1.10 | 1.84/1.32 | 2.02/1.39 | 2.58/1.86 | 2.04/1.56 |

**Brecha frente al paper, h = 1–15 (réplica − paper; `results/exp1_brecha_vs_paper.csv`)**

| Modelo | MASE media | MASE mediana | RMSSE media | RMSSE mediana |
|---|---|---|---|---|
| FTGM_OLS_paper vs FTGM | +3.18 | +2.26 | +2.64 | +2.25 |
| FTGM_OLS_refit vs FTGM | +1.64 | +1.27 | +1.42 | +1.33 |
| FTGM_Ridge vs FTGM | +1.50 | +1.23 | +1.26 | +1.11 |
| SNaive vs SNaive | +0.05 | +0.05 | +0.02 | +0.03 |
| ETS vs HWES | −0.04 | +0.08 | −0.16 | +0.02 |
| SARIMA vs SARIMA | −0.16 | −0.30 | −0.23 | −0.18 |

**Comparación con el naive estacional por serie, h = 1–15 (`results/exp1_victorias_vs_snaive.csv`; Wilcoxon de
rangos con signo pareado, n = 70)**

| Modelo | % series con MASE < SNaive | p (MASE) | % series con RMSSE < SNaive | p (RMSSE) |
|---|---|---|---|---|
| FTGM_OLS_paper | 28.6 | 0.000009 | 25.7 | 0.000002 |
| FTGM_OLS_refit | 44.3 | 0.177 | 42.9 | 0.122 |
| FTGM_Ridge | 50.0 | 0.750 | 47.1 | 0.858 |
| FTGM_OLS_ancla | 44.3 | 0.262 | 44.3 | 0.172 |
| Motor_torneo | 47.1 | 0.557 | 50.0 | 0.595 |
| Naive | 52.9 | 0.229 | 51.4 | 0.330 |
| ETS | 62.9 | 0.0012 | 64.3 | 0.0021 |
| SARIMA | 67.1 | 0.0014 | 65.7 | 0.0013 |

Primer lugar por RMSSE 1–15 entre FTGM_OLS_paper, SNaive, Naive, ETS y SARIMA: SARIMA 34 series, SNaive 10,
FTGM_OLS_paper 9, Naive 9, ETS 8 (`results/exp1_primer_lugar_rmsse.csv`). En el paper, el FTGM ocupó el
primer lugar en 38 de 70 series frente a 11 modelos.

Otros registros (`results/exp1_modelos_elegidos.csv`):
- Orden de Fourier elegido por el Algoritmo 1 (MCO, 420 casos): N = 1: 254; N = 2: 87; N = 3: 24; N = 4: 15;
  N = 5: 40. Con ridge: N = 1: 118; 2: 116; 3: 58; 4: 20; 5: 108, y λ = 0: 60; 0.5: 70; 3: 67; 10: 223.
  Ninguna integración divergió (0 de 420 en las cuatro variantes del FTGM).
- Ganador del torneo del motor (420 orígenes): DampedTrend 113, SeasonalDamped 101, FTGM 83, FTGMCombo 76,
  SeasonalNaive 47 (familia FTGM: 159 de 420).
- ETS: (A,N,N) en 321 de 420 orígenes. SARIMA: d = 1 en 155 de 420.
- **Verificación numérica de la EDO** (`results/exp1_verificacion_edo.csv`): el ajuste y el pronóstico del
  motor (RK45, rtol 1e-6) se compararon con DOP853 (rtol 1e-11) en 350 ajustes (70 series × N = 1…5); 350
  fueron estables y la diferencia máxima relativa al nivel medio fue 3.66e-4 en el ajuste y 2.65e-3 en el
  pronóstico.

Figuras: `figures/fig_exp1_rmsse_boxplot.png` (RMSSE 1–15 por serie; el número sobre cada caja es la
mediana) y `figures/fig_exp1_mase_horizonte.png` (media y mediana de MASE por horizonte).

### Interpretación
1. **La exactitud que reporta el paper para el FTGM no se reprodujo.** Con la lectura literal del
   Algoritmo 1 (FTGM_OLS_paper), la MASE media 1–15 es 4.83 frente a 1.65 del paper; con re-estimación en
   todo el in-sample, 3.29; con ridge, 3.15. En las tres variantes el FTGM queda por detrás del SNaive en
   la media y no supera al SNaive de forma significativa en ninguna (p ≥ 0.12); la variante literal es
   significativamente peor que el SNaive (p < 0.0001).
2. **La brecha no proviene de los datos ni del protocolo de evaluación.** Con la misma construcción mensual
   y las mismas métricas, el SNaive, el ETS y el SARIMA quedan a ≤ 0.30 del paper en la media y la mediana
   de MASE y RMSSE 1–15; el SNaive, que no tiene parámetros, queda a 0.05.
3. **La brecha tampoco proviene de la estimación ni del integrador del motor.** Los parámetros de la
   Tabla C.1 se reproducen en 3–4 cifras significativas, la integración coincide con un integrador de alta
   precisión (diferencia ≤ 0.27 % del nivel) y anclar la EDO a la suma acumulada observada casi no cambia
   el resultado (MASE 3.16 frente a 3.29). Los pronósticos del FTGM amplifican la estacionalidad estimada en
   el tramo de entrenamiento (por ejemplo, Foods_1_CA_2, origen 43: pronóstico de 23 058 unidades para un
   mes con 13 859 reales; `results/exp1_pronosticos_crudos.csv`).
4. La causa de la diferencia queda sin identificar con la información disponible: el repositorio de los
   autores (`docs/ftgmmodel/utils/ftgm_*.m`) no incluye la función `odefcn` que integra el modelo ni el
   script de evaluación, y el código MATLAB usa una estimación en dos etapas con un valor inicial `eta` que
   no se describe en el artículo. Esta limitación debe declararse al citar la Tabla 1.
5. El **pipeline de producción** (Motor_torneo), que pone al FTGM a competir con suavizados y el naive
   estacional, obtiene MASE 1–15 de 2.65/2.26, mejor que cualquier FTGM aislado y que el SNaive en la media,
   pero sin diferencia significativa con el SNaive (p = 0.56) y por detrás del ETS (2.43) y del SARIMA
   (2.28). El torneo eligió un modelo de la familia FTGM en 159 de 420 orígenes.
6. Una serie (Foods_2_CA_2) tiene un cambio de nivel que produce RMSSE de 17.05 a 17.72 en todos los
   modelos; es el valor atípico común de la Figura de cajas.

### Limitaciones
- ETS y SARIMA de `statsmodels` no son idénticos a `ets()` y `auto.arima()` de R (búsqueda, criterios de
  admisión y pruebas de raíz unitaria difieren); por eso se usan solo como referencias de orden de magnitud.
- No se replicaron SGM, DGGM, DGSM, DHR-ARIMA, BATS, TBATS, MLP ni LSTM.
- En la Tabla 1 del paper, la columna h = 12 del SNaive (3.75/3.25) coincide con el valor de la réplica en
  h = 13 (3.79/3.27) y no con h = 12 (2.84/2.01); en los demás horizontes la réplica del SNaive queda a
  ≤ 0.13 en MASE. No se puede determinar si se trata de un desfase en la tabla del paper.
- Las RMSSE por horizonte del SNaive de la réplica quedan 0.05–0.27 por debajo del paper (salvo h = 12);
  la definición exacta con la que el paper agrega la RMSSE por horizonte no se especifica.

### Comando y tiempo
`python exp1_replicacion_ye.py` — 7 min 29 s de reloj (16 procesos). Log: `logs/exp1_log.txt`.

---

## Experimento 2 — Diagnósticos de modelo gris (pedido del asesor)

### Objetivo
Evaluar la calidad del ajuste dentro de muestra del FTGM con las pruebas clásicas de la teoría de sistemas
grises.

### Datos y protocolo
Las mismas 70 series de 63 meses del Experimento 1. Para cada serie se elige el orden con el Algoritmo 1
(validación de 6 meses, N ≤ 5) y se re-estima con las 63 observaciones: FTGM_OLS (λ = 0, paper) y
FTGM_Ridge (rejilla del motor). Como referencia se evalúa el ajuste del naive estacional (x̂_k = x_{k−12},
k ≥ 13). Indicadores (`common.grey_diagnostics`):

- C = S2/S1 (desviación estándar de los residuos / desviación estándar de los datos);
- p = P(|e_k − ē| < 0.6745·S1);
- error relativo medio = media de |e_k|/x_k (%);
- grado relacional de Deng (ρ = 0.5) entre ajuste y real, con ambas secuencias divididas por la media de
  los datos reales.

**Grados de precisión.** Se usan los umbrales de la Tabla 2 de Liu, Dang, Lv, Zhao, Zhu y Wang (2025),
"Forecasting antimicrobial resistance in China using a hybrid ARIMA-GM(1,1) model", *BMC Infectious
Diseases*, 25, 1020, https://doi.org/10.1186/s12879-025-11483-4 (DOI verificado en Crossref; tabla leída
en el texto completo de PMC12355814): nivel I *excelente* C < 0.35 y P ≥ 0.95; nivel II *bueno*
C < 0.50 y P ≥ 0.80; nivel III *calificado* C < 0.65 y P ≥ 0.70; nivel IV *no calificado* en otro caso.
Se asigna el mejor nivel cuyas dos condiciones se cumplen. Nota: los nombres de esa fuente
(excelente/bueno/calificado/no calificado) difieren de los usados en el pedido (bueno/calificado/apenas
calificado/no calificado); los umbrales son los mismos.

### Resultados (`results/exp2_*.csv`, `logs/exp2_log.txt`)

| Indicador (70 series) | FTGM_OLS media / mediana [mín–máx] | FTGM_Ridge media / mediana | SNaive ref. media / mediana |
|---|---|---|---|
| C | 0.609 / 0.554 [0.222–0.973] | 0.627 / 0.618 | 1.072 / 1.015 |
| p | 0.735 / 0.746 [0.413–1.000] | 0.722 / 0.738 | 0.498 / 0.461 |
| Error relativo medio (%) | 13.06 / 10.70 [4.54–89.96] | 13.41 / 10.78 | 17.56 / 17.46 |
| Grado relacional (ρ = 0.5) | 0.665 / 0.665 [0.522–0.795] | 0.668 / 0.665 | 0.666 / 0.660 |

| Grado | FTGM_OLS | FTGM_Ridge | SNaive ref. |
|---|---|---|---|
| I Excelente | 9 | 8 | 0 |
| II Bueno | 18 | 14 | 3 |
| III Calificado | 10 | 15 | 7 |
| IV No calificado | 33 | 33 | 60 |

Grado relacional ≥ 0.6: 64 de 70 series (FTGM_OLS), 63 (FTGM_Ridge) y 60 (SNaive). Orden elegido con
63 observaciones (MCO): N = 1 en 33 series, 2 en 16, 3 en 4, 4 en 5, 5 en 12. Ninguna divergencia.
Figura: `figures/fig_exp2_grados_gris.png`.

### Interpretación
37 de 70 series (52.9 %) alcanzan un grado I–III con el FTGM_OLS y 33 (47.1 %) quedan como no calificadas.
El ajuste dentro de muestra del FTGM es más cercano a los datos que el del naive estacional (C mediana 0.554
frente a 1.015). El grado relacional apenas distingue entre modelos (0.665 frente a 0.660), por lo que aporta
poca información para este fin. Estas pruebas miden el **ajuste dentro de muestra**; el Experimento 1 muestra
que un buen ajuste no se tradujo en buena exactitud fuera de muestra.

### Limitaciones
El grado relacional depende de la normalización elegida (aquí, división por la media); con otras
normalizaciones cambia de valor. La referencia SNaive se evalúa en 51 observaciones (k ≥ 13) y el FTGM en 63.

### Comando y tiempo
`python exp2_diagnostico_gris.py` — 14 s. Log: `logs/exp2_log.txt`.

---

## Experimento 3 — Series cortas tipo MYPE con el pipeline de producción (RQ2)

### Objetivo
Medir, con el mismo contrato que usa la API, qué modelo elige el motor y con qué error, según la historia
disponible, y compararlo con referencias simples.

### Datos
- Población elegible: productos-tienda del M5 con venta media diaria ≥ 0.2 en los 24 meses previos al
  corte y con al menos una venta antes del inicio de la ventana de 36 meses: 17 863 de 30 490.
- Muestra: 300 series, estratificación proporcional por categoría, semilla 42: FOODS 161, HOUSEHOLD 93,
  HOBBIES 46 (`results/exp3_muestra.csv`). Mediana de la venta media diaria: FOODS 0.95, HOUSEHOLD 0.67,
  HOBBIES 0.46 unidades.

### Protocolo
- Corte común `as_of` = 2016-02-01 (lunes y primer día de mes). Historias truncadas a 6, 9, 12, 18 y 24
  meses (y 36 como referencia adicional), que terminan el 2016-01-31.
- Puntos diarios dispersos (solo días con venta), como los arma la API (`erp_analysis.build_points`).
- `ForecastService().forecast(ForecastRequest(frequency="auto", horizon_days=H, as_of=2016-02-01))` con
  H = 30 y 90.
- Evaluación sobre los periodos que devuelve el motor; el valor real de cada periodo es la suma de ventas
  del M5 en ese mes o semana. Con frecuencia mensual, 30 días equivalen a 1 mes (feb-2016, 29 días) y 90
  días a 3 meses (90 días); con frecuencia semanal, a 5 semanas (35 días) y 13 semanas (91 días).
- Referencias sobre los **mismos periodos** y la misma historia sin reparar ni limpiar: Naive (último
  periodo completo), SNaive (mismo periodo de la temporada anterior; si la historia es menor que una
  temporada, equivale al Naive) y, en modo mensual, FTGM solo (Algoritmo 1 con la rejilla ridge del motor,
  N ≤ min(5, round(2n/12)), sin torneo; si falla se usa SNaive: 2 fallos en 24 meses y 3 en 36).
- Métricas: WAPE del total del horizonte (Σ|total pronosticado − total real| / Σ total real), WAPE por
  periodo, MAE, mediana del error porcentual absoluto del total por producto (MdAPE) y % de productos con
  sobrepronóstico extremo (total pronosticado > 3 veces el real). Prueba de Wilcoxon de rangos con signo,
  pareada por producto, sobre el MAE por periodo y sobre el error absoluto del total.
- El motor omitió (`status = skipped`) 4 productos con 6 meses y 2 con 9 meses porque no tuvieron ventas en
  esa ventana; los porcentajes de esas filas se calculan sobre 296 y 298 productos.

### Resultados

**Frecuencia elegida por el motor** (`results/exp3_frecuencia_por_largo.csv`): semanal en todos los productos
con 6, 9, 12 y 18 meses; con 24 meses, mensual en 262 y semanal en 38 (su primera venta de la ventana cae
después del primer mes, por lo que tienen menos de 24 meses completos); con 36 meses, mensual en 298.

**Ganador del torneo del motor, % de productos** (`results/exp3_ganador_por_largo.csv`)

| H | Historia (meses) | Familia FTGM | Croston-SBA | Promedio móvil | Suav. amortiguado | Suav. estacional | Naive estacional |
|---|---|---|---|---|---|---|---|
| 30 | 6 | 36.1 | 4.7 | 11.8 | 31.1 | 0.0 | 16.2 |
| 30 | 9 | 58.4 | 3.7 | 3.7 | 21.8 | 0.0 | 12.4 |
| 30 | 12 | 65.3 | 4.0 | 2.3 | 17.0 | 0.0 | 11.3 |
| 30 | 18 | 69.0 | 4.3 | 0.7 | 16.3 | 0.0 | 9.7 |
| 30 | 24 | 60.0 | 2.0 | 0.0 | 22.0 | 0.0 | 16.0 |
| 30 | 36 | 51.7 | 1.3 | 0.0 | 20.7 | 21.3 | 5.0 |
| 90 | 6 | 27.7 | 4.7 | 11.8 | 33.8 | 0.0 | 22.0 |
| 90 | 9 | 51.0 | 3.7 | 3.7 | 26.8 | 0.0 | 14.8 |
| 90 | 12 | 60.7 | 4.0 | 2.3 | 21.7 | 0.0 | 11.3 |
| 90 | 18 | 70.7 | 4.3 | 0.7 | 16.3 | 0.0 | 8.0 |
| 90 | 24 | 58.0 | 2.0 | 0.0 | 19.3 | 0.0 | 20.7 |
| 90 | 36 | 49.7 | 1.3 | 0.0 | 22.7 | 17.7 | 8.7 |

Figura: `figures/fig_exp3_ganador_por_largo.png`.

**Error del pipeline frente a referencias** (`results/exp3_error_por_largo.csv`)

| H | Hist. | Método | WAPE total % | WAPE periodo % | MdAPE total % | % sobrepron. ×3 |
|---|---|---|---|---|---|---|
| 30 | 6 | Motor | 39.70 | 50.14 | 30.03 | 6.8 |
| 30 | 6 | Naive = SNaive | 38.24 | 46.47 | 40.00 | 3.4 |
| 30 | 9 | Motor | 35.09 | 47.67 | 30.16 | 6.4 |
| 30 | 9 | Naive = SNaive | 38.30 | 46.51 | 40.15 | 3.4 |
| 30 | 12 | Motor | 44.27 | 56.34 | 33.52 | 6.7 |
| 30 | 12 | Naive | 38.30 | 46.51 | 40.15 | 3.3 |
| 30 | 12 | SNaive | 45.25 | 62.09 | 37.78 | 7.0 |
| 30 | 18 | Motor | 40.68 | 52.50 | 34.47 | 6.3 |
| 30 | 18 | Naive | 38.30 | 46.51 | 40.15 | 3.3 |
| 30 | 18 | SNaive | 48.30 | 65.23 | 38.26 | 7.3 |
| 30 | 24 | Motor | 42.19 | 42.89 | 33.19 | 8.7 |
| 30 | 24 | Naive | 32.32 | 34.36 | 29.41 | 3.3 |
| 30 | 24 | SNaive | 52.36 | 54.25 | 40.00 | 8.7 |
| 30 | 24 | FTGM solo (n = 262) | 52.08 | 52.08 | 35.20 | 9.9 |
| 30 | 36 | Motor | 45.67 | 45.67 | 31.88 | 9.7 |
| 30 | 36 | Naive | 32.10 | 32.20 | 29.15 | 3.7 |
| 30 | 36 | SNaive | 52.74 | 52.84 | 40.00 | 8.7 |
| 30 | 36 | FTGM solo (n = 298) | 48.64 | 48.64 | 35.15 | 11.7 |
| 90 | 6 | Motor | 43.75 | 58.40 | 32.55 | 6.1 |
| 90 | 6 | Naive = SNaive | 44.62 | 55.01 | 40.79 | 3.7 |
| 90 | 9 | Motor | 39.10 | 55.45 | 30.49 | 4.7 |
| 90 | 9 | Naive = SNaive | 44.83 | 55.18 | 40.82 | 3.7 |
| 90 | 12 | Motor | 43.01 | 58.11 | 33.02 | 5.7 |
| 90 | 12 | Naive | 44.91 | 55.24 | 40.91 | 3.7 |
| 90 | 12 | SNaive | 40.87 | 62.94 | 33.33 | 4.3 |
| 90 | 18 | Motor | 49.45 | 64.24 | 29.95 | 6.7 |
| 90 | 18 | Naive | 44.91 | 55.24 | 40.91 | 3.7 |
| 90 | 18 | SNaive | 42.98 | 66.10 | 31.43 | 4.0 |
| 90 | 24 | Motor | 45.08 | 52.68 | 29.55 | 4.7 |
| 90 | 24 | Naive | 36.59 | 42.61 | 27.94 | 3.0 |
| 90 | 24 | SNaive | 45.41 | 55.03 | 31.82 | 4.7 |
| 90 | 24 | FTGM solo (n = 262) | 60.95 | 66.18 | 29.55 | 4.6 |
| 90 | 36 | Motor | 43.88 | 49.14 | 27.40 | 5.3 |
| 90 | 36 | Naive | 37.07 | 41.91 | 28.00 | 3.3 |
| 90 | 36 | SNaive | 45.62 | 54.05 | 31.71 | 4.7 |
| 90 | 36 | FTGM solo (n = 298) | 46.00 | 51.50 | 33.08 | 5.0 |

El MAE en unidades está en el CSV. Figura: `figures/fig_exp3_wape_por_largo.png`.

**Comparaciones pareadas por producto** (`results/exp3_pruebas_pareadas.csv`; "gana" = menor MAE por periodo)

| H | Hist. | Comparación | n | Motor gana % | Empate % | Motor pierde % | p Wilcoxon (MAE periodo) | p Wilcoxon (error total) |
|---|---|---|---|---|---|---|---|---|
| 30 | 6 | Motor vs SNaive (= Naive) | 296 | 47.0 | 16.2 | 36.8 | 0.207 | 0.114 |
| 30 | 9 | Motor vs SNaive (= Naive) | 298 | 50.3 | 12.4 | 37.2 | 0.024 | 0.014 |
| 30 | 12 | Motor vs SNaive | 300 | 62.0 | 8.7 | 29.3 | 2.1e-10 | 0.054 |
| 30 | 12 | Motor vs Naive | 300 | 48.3 | 6.3 | 45.3 | 0.641 | 0.557 |
| 30 | 18 | Motor vs SNaive | 300 | 63.3 | 7.7 | 29.0 | 1.3e-11 | 0.022 |
| 30 | 18 | Motor vs Naive | 300 | 48.7 | 5.3 | 46.0 | 0.962 | 0.178 |
| 30 | 24 | Motor vs SNaive | 300 | 53.0 | 16.0 | 31.0 | 3.4e-07 | 3.4e-06 |
| 30 | 24 | Motor vs Naive | 300 | 42.0 | 3.0 | 55.0 | 0.0062 | 0.0062 |
| 30 | 24 | Motor vs FTGM solo | 262 | 53.8 | 0.4 | 45.8 | 0.172 | 0.172 |
| 90 | 6 | Motor vs SNaive (= Naive) | 296 | 42.9 | 21.3 | 35.8 | 0.110 | 0.0070 |
| 90 | 9 | Motor vs SNaive (= Naive) | 298 | 48.3 | 12.4 | 39.3 | 0.0037 | 0.00022 |
| 90 | 12 | Motor vs SNaive | 300 | 65.3 | 6.3 | 28.3 | 7.8e-14 | 0.679 |
| 90 | 12 | Motor vs Naive | 300 | 51.0 | 3.0 | 46.0 | 0.034 | 0.0059 |
| 90 | 18 | Motor vs SNaive | 300 | 68.0 | 3.7 | 28.3 | 1.2e-10 | 0.460 |
| 90 | 18 | Motor vs Naive | 300 | 49.0 | 2.0 | 49.0 | 0.584 | 0.066 |
| 90 | 24 | Motor vs SNaive | 300 | 52.7 | 19.7 | 27.7 | 0.00012 | 0.044 |
| 90 | 24 | Motor vs Naive | 300 | 43.0 | 1.7 | 55.3 | 0.0087 | 0.152 |
| 90 | 24 | Motor vs FTGM solo | 262 | 53.8 | 0.4 | 45.8 | 0.275 | 0.014 |

Las filas de 36 meses están en el CSV. La prueba de Wilcoxon es bilateral: el sentido de una diferencia
significativa se lee en las columnas "gana/pierde" y en las tablas de WAPE. No se aplicó el contraste de
Diebold-Mariano porque cada producto aporta un solo origen de pronóstico; la comparación es transversal
(entre productos).

### Interpretación
1. **Selección de modelo.** La familia FTGM gana el torneo en 27.7–36.1 % de los productos con 6 meses y en
   51.0–70.7 % con 9 a 24 meses; Croston-SBA se usa en 1.3–4.7 % y el promedio móvil en 11.8 % con 6 meses
   y en ≤ 3.7 % desde 9 meses. Con 36 meses aparece el suavizado estacional (17.7–21.3 %). Con historias
   de 6 a 18 meses el motor trabaja en frecuencia **semanal** (T = 52), con menos de dos temporadas.
2. **Frente al naive estacional**, el motor tiene menor MAE por periodo en la mayoría de los productos con
   12 a 36 meses de historia (gana en 52.7–68.0 % y pierde en 27.7–35.3 %; p ≤ 0.00025). Con 6 y 9 meses
   el naive estacional equivale al naive simple porque la historia no cubre una temporada.
3. **Frente al naive simple (último periodo)**, el motor **no** muestra ventaja: con 24 meses pierde en
   55.0–55.3 % de los productos (p ≤ 0.0087) y su WAPE por periodo es mayor en todos los largos de
   historia (por ejemplo, 42.89 % frente a 34.36 % con 24 meses y H = 30). En la mediana por producto
   (MdAPE del total) el motor es menor que el Naive con 6–18 meses (29.95–34.47 % frente a 40.00–40.91 %),
   pero el motor genera más sobrepronósticos extremos (4.7–9.7 % de productos frente a 3.0–3.7 %), que
   elevan el WAPE agregado.
4. **FTGM solo frente al motor**: con 24 meses, el WAPE del total es 52.08 % para el FTGM solo
   (262 productos en modo mensual) y 42.19 % para el motor (300 productos) con H = 30, y 60.95 % frente a
   45.08 % con H = 90. En la comparación pareada sobre los 262 productos comunes, el motor gana en 53.8 %;
   la diferencia es significativa solo en el error total con H = 90 (p = 0.014).
5. La afirmación de `FUENTE_DE_VERDAD.md` §5 sobre dónde gana la familia FTGM proviene de datos de
   demostración generados programáticamente; en el M5 los porcentajes son distintos (por ejemplo, 36.1 %
   con 6 meses frente a ~10 %), por lo que no deben mezclarse.

### Sub-experimento 3b — Censura por quiebre **inyectada de forma controlada sobre datos M5**

**Protocolo.** En 100 de los 300 productos (elegidos al azar, semilla 42), con historias de 12 y 24 meses,
se ponen en cero L/4 ventanas de 7 a 14 días (3 ventanas con 12 meses y 6 con 24; separadas por ≥ 7 días;
media de 31.8 y 64.0 días censurados; 5 322 y 12 183 unidades eliminadas en total). Condiciones:
A = historia original; B = historia censurada sin marcar (los ceros se toman como demanda);
C = historia censurada con `stockout_flag = True` en los días afectados. El error se mide contra las ventas
reales futuras (no censuradas). También se mide la distorsión de la historia que ve el modelo:
Σ|serie limpiada de B o C − serie limpiada de A| / Σ serie limpiada de A.

**Resultados** (`results/exp3_censura_resumen.csv`, `results/exp3_censura_pruebas.csv`)

| H | Hist. | Condición | WAPE total % | WAPE periodo % | Sesgo total % | MdAPE total % | Distorsión historia % (media) |
|---|---|---|---|---|---|---|---|
| 30 | 12 | A original | 59.81 | 73.29 | +17.25 | 38.03 | 0.00 |
| 30 | 12 | B sin marcar | 42.50 | 57.54 | +5.53 | 34.05 | 9.82 |
| 30 | 12 | C marcada | 56.28 | 71.06 | +24.35 | 33.05 | 8.21 |
| 30 | 24 | A original | 45.63 | 46.21 | +2.41 | 36.58 | 0.00 |
| 30 | 24 | B sin marcar | 43.52 | 43.97 | −4.72 | 36.53 | 12.44 |
| 30 | 24 | C marcada | 43.94 | 44.65 | −0.04 | 35.54 | 16.83 |
| 90 | 12 | A original | 42.76 | 61.10 | +7.57 | 37.08 | 0.00 |
| 90 | 12 | B sin marcar | 72.52 | 84.71 | +17.61 | 36.91 | 9.82 |
| 90 | 12 | C marcada | 40.89 | 61.69 | +12.02 | 35.20 | 8.21 |
| 90 | 24 | A original | 50.50 | 60.38 | +10.67 | 30.88 | 0.00 |
| 90 | 24 | B sin marcar | 48.56 | 55.27 | −1.73 | 30.77 | 12.44 |
| 90 | 24 | C marcada | 51.05 | 58.80 | +7.16 | 35.29 | 16.83 |

| H | Hist. | n | C con menor error que B (% productos) | p Wilcoxon (error) | Historia de C más fiel que la de B (% productos) | p Wilcoxon (historia) |
|---|---|---|---|---|---|---|
| 30 | 12 | 100 | 57.0 | 0.262 | 66.0 | 0.00054 |
| 30 | 24 | 100 | 45.0 | 0.826 | 72.0 | 0.000032 |
| 90 | 12 | 100 | 49.0 | 0.858 | 66.0 | 0.00054 |
| 90 | 24 | 100 | 48.0 | 0.658 | 72.0 | 0.000032 |

Unidades recuperadas por la reparación (serie limpiada de C menos la de B, sumadas sobre los 100 productos):
5 294 de 5 322 unidades censuradas con 12 meses (99.5 %) y 13 717.6 de 12 183 con 24 meses (112.6 %,
sobrerreparación de 1 534.6 unidades) (`logs/verificaciones_log.txt`).
Figura: `figures/fig_exp3_censura_ablacion.png`.

**Interpretación.**
1. Marcar el quiebre **sí corrige la historia**: en 66–72 % de los productos la serie reparada (C) queda
   más cerca de la serie sin censura que la serie sin marcar (B) (p ≤ 0.00054), y en el agregado recupera
   el 99.5 % (12 meses) y el 112.6 % (24 meses) de las unidades eliminadas. Con 24 meses la distorsión media
   de C (16.83 %) supera a la de B (12.44 %) aunque la mayoría de productos mejora: hay sobrerreparación en
   algunos productos (el escalado mensual x/(1 − s) con tope de 2 veces el valor interpolado).
2. Esa mejora **no se trasladó a un error de pronóstico significativamente menor**: C supera a B en
   45–57 % de los productos y ninguna diferencia es significativa (p ≥ 0.26).
3. El ruido de la selección de modelo domina el resultado: entre A y B cambia el modelo ganador en
   31.0–48.0 % de los productos y entre A y C en 29.0–41.0 % (`logs/verificaciones_log.txt`), y algunas
   ejecuciones del FTGM semanal generan sobrepronósticos de un orden de magnitud (por ejemplo, un producto
   con 75 unidades reales y 2 543 pronosticadas en B con 12 meses y H = 90). Por ello B puede tener menor
   WAPE que A, como ocurre con 12 meses y H = 30, o mayor, como ocurre con 12 meses y H = 90.

### Limitaciones del Experimento 3
- Un solo corte (feb-2016); los resultados pueden depender de la estación del corte.
- El M5 no registra quiebres reales; la censura del 3b es sintética y uniforme en el tiempo.
- Las series son de baja rotación (mediana de 0.46–0.95 unidades diarias por categoría); los porcentajes
  de error son altos para todos los métodos.
- El WAPE agregado es sensible a pocos productos con sobrepronóstico extremo; por eso se reportan también
  la MdAPE y el porcentaje de casos con sobrepronóstico ×3.

### Comando y tiempo
`python exp3_series_cortas.py` — 1 min 13 s (16 procesos; 4 200 llamadas al motor). Log: `logs/exp3_log.txt`.

---

## Experimento 4 — Pruebas automatizadas y latencia

### Protocolo
1. `python -m pytest -q` en `inventory-dss-ftgm-engine`, con `PYTHONPATH` apuntando al motor. Salida
   literal: `logs/exp4_pytest.txt`.
2. Latencia de `ForecastService().forecast()` para **un** producto, en el mismo proceso (sin HTTP), con el
   contrato de la API (`frequency="auto"`, `as_of=2016-02-01`). Producto fijo: el primero de la muestra
   del Exp. 3 (`HOBBIES_1_159_CA_1_evaluation`). 3 llamadas de calentamiento y 50 medidas, con la máquina
   sin otras corridas.

### Resultados
- **pytest: 40 pruebas superadas en 4.12 s** (código de salida 0).
- Latencia (`results/exp4_latencia.csv`):

| Historia | H | Frecuencia | Ganador | Puntos de entrada | Mediana (ms) | p90 (ms) | Mín–máx (ms) |
|---|---|---|---|---|---|---|---|
| 12 meses | 30 días | semanal | DampedTrend | 168 | 51.5 | 54.2 | 48.2–77.7 |
| 12 meses | 90 días | semanal | DampedTrend | 168 | 49.5 | 56.8 | 46.9–71.0 |
| 24 meses | 30 días | mensual | FTGM | 252 | 72.9 | 75.4 | 70.4–90.3 |
| 24 meses | 90 días | mensual | FTGMCombo | 252 | 71.2 | 83.6 | 68.9–109.2 |

En el Exp. 3 (16 procesos en paralelo, 300 productos × 6 historias × 2 horizontes) la mediana por llamada
fue 0.152 s y el percentil 90, 0.316 s (`logs/exp3_log.txt`).

### Limitaciones
La latencia excluye la red, la serialización HTTP y la base de datos de la API; corresponde solo al cómputo
del motor en un equipo de escritorio de 16 núcleos lógicos.

### Comando y tiempo
`python exp4_pruebas.py` — 23 s. Log: `logs/exp4_log.txt`.

---

## Experimento 7 — Simulación de la decisión de reposición (RQ3 del paper JISEBI; en el paper es el Exp. 4)

**Script:** `exp7_simulacion_decision.py` (07/11/2026). **Resultados:** `results/exp7_*.csv`, `logs/exp7_log.txt`.

### Objetivo
Evaluar la capa de decisión del DSS: con la misma regla de pedido del sistema, ¿qué pronóstico deja mejores decisiones de inventario?

### Protocolo
- **Productos:** los mismos 300 productos-tienda del Exp. 3 (muestra con semilla 42).
- **Revisiones:** seis, el primer día de cada mes, del 2015-09-01 al 2016-02-01. La simulación diaria va del 2015-09-01 al 2016-02-29 (182 días).
- **Historia:** la inicial es de 6, 9, 12, 18 o 24 meses antes del 2015-09-01 y crece con cada revisión.
- **Motor:** servicio de producción (`ForecastService`, frecuencia automática, horizonte de 90 días) sobre una copia limpia del commit `be6a98f` (`git archive`; la misma revisión de los Exp. 1–3). Se comprobó que reproduce exactamente los pronósticos guardados del Exp. 3 (diferencia máxima 0 en 10 casos). La línea «con cambios locales sin commit» del log se refiere al repositorio de trabajo, no a la copia usada.
- **Regla del sistema (ecuación 7 del paper):**
  - nivel objetivo O = D(L+R), con plazo L = 7 días y revisión R = 30 días;
  - pedido Q = ⌈k·O − posición⌉;
  - k = 1 es la regla sin stock de seguridad. Para trazar la frontera servicio–inventario se usa k ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 3}.
- **Dinámica:** ventas perdidas, pedido que llega 7 días después y stock inicial ⌈k·O⌉.
- **Referencias:**
  - Naive y SNaive, sobre los mismos periodos y la misma historia cruda;
  - Oráculo: la demanda real de los 37 días siguientes, como cota superior de la regla.
- **Revisiones omitidas por el motor:** 51 de 9 000 (*skipped*). En ellas el sistema no recomienda, así que se simulan con demanda 0.
- **Métricas:**
  - tasa de servicio agregada (atendido / demanda) y su mediana por producto;
  - ventas perdidas;
  - días de inventario (stock promedio / demanda diaria promedio).
- **Pruebas:** Wilcoxon pareadas por producto, 30 pruebas con corrección de Holm.

### Resultados (k = 1, `results/exp7_resumen.csv` y `exp7_pruebas_pareadas.csv`)

| Historia | Motor: servicio / días de inv. | Naive | SNaive | Oráculo |
|---|---|---|---|---|
| 6 | 83.1 % / 24.0 | 79.1 % / 23.2 | = Naive | 98.5 % / 15.8 |
| 9 | 82.1 % / 23.9 | 79.1 % / 23.2 | 80.4 % / 23.4 | 98.5 % / 15.8 |
| 12 | 82.6 % / 22.0 | 79.1 % / 23.2 | 78.8 % / 21.3 | 98.5 % / 15.8 |
| 18 | 80.7 % / 20.0 | 79.1 % / 23.2 | 77.3 % / 20.9 | 98.5 % / 15.8 |
| 24 | 80.3 % / 20.7 | 85.4 % / 23.2 | 74.7 % / 19.8 | 98.5 % / 15.8 |

- **Frente al Naive, con 6 a 18 meses:** el motor da más servicio en el 59–64 % de los productos (p de Holm ≤ 0.034) y pierde menos ventas (p de Holm ≤ 0.013), sin más inventario. Con 12 y 18 meses incluso usa menos (p de Holm 0.026 y 1.8e-5).
- **Con 24 meses (modo mensual):** el Naive tiene más servicio agregado (85.4 % frente a 80.3 %; por producto no es significativo tras Holm, p = 0.10), pero el motor usa menos inventario (p de Holm 0.0049).
- **Frente al SNaive:**
  - con 9 meses, el motor da más servicio (p de Holm 0.013);
  - con 12 y 18 meses no hay diferencia significativa;
  - con 24 meses, el motor da más servicio (p de Holm 0.040).
- **Frontera** (`results/exp7_frontera.csv`), días de inventario para un 90 % de servicio agregado:
  - motor: 37.1 / 40.4 / 34.4 / 33.4 / 37.7 (6 / 9 / 12 / 18 / 24 meses);
  - Naive: 43.3 en 6–18 meses y 30.2 con 24 meses;
  - SNaive: 40.5–71.3;
  - Oráculo: 13.1.
- **Lectura:** con 6 a 18 meses de historia, el motor alcanza el mismo servicio con menos inventario que las referencias simples. Con 24 meses (modo mensual), el naive es mejor. La distancia al oráculo (13 días frente a 33–40) muestra cuánto valor queda en mejorar el pronóstico.

### Limitaciones
- Datos de un gran minorista (M5), no de MYPE.
- Sin costos monetarios, por lo que se reporta la frontera servicio–inventario.
- Plazo de entrega fijo de 7 días.
- Un solo periodo de 6 meses (septiembre a febrero, con la campaña de fin de año).
- k es un factor proporcional, no el stock de seguridad que configura el usuario.

### Comando y tiempo
`ENGINE_DIR=<copia de be6a98f> ENGINE_COMMIT=be6a98f… python exp7_simulacion_decision.py`: 171 s con 16 procesos.

### Corrección de Holm de los Exp. 1, 3 y 3b (07/11/2026)
- **Scripts:** `exp3_holm.py`, que también ajusta el Exp. 1 y la censura.
- **Salidas:** `results/exp3_pruebas_pareadas_holm.csv` (48 pruebas), `exp1_victorias_vs_snaive_holm.csv` (16) y `exp3_censura_pruebas_holm.csv` (8).
- **Exp. 1:** ETS y SARIMA siguen siendo significativamente mejores que SNaive (p de Holm ≤ 0.023), y el FTGM literal, significativamente peor.
- **Exp. 3:** el motor supera a SNaive en error por periodo con 12–36 meses en ambos horizontes (p de Holm ≤ 0.0094). En error del total solo lo supera con 24–36 meses a 30 días; a 90 días con 12–18 meses no hay diferencia (p = 0.68 y 0.46 sin corregir). Frente al naive no hay ninguna ventaja significativa; con 36 meses el naive es mejor (p de Holm 0.019).
- **Censura:** la historia reparada es más fiel (p de Holm ≤ 0.0033), y el error del pronóstico no difiere (p de Holm = 1).

## Síntesis para la memoria

- **RQ1 (fidelidad):** la estimación y la integración del motor reproducen las ecuaciones del paper
  (Tabla C.1 en 3–4 cifras; EDO con diferencia ≤ 0.27 %), pero la exactitud fuera de muestra reportada
  en la Tabla 1 **no** se reprodujo (MASE 1–15: 3.29 con re-estimación y 4.83 con la lectura literal del
  Algoritmo 1, frente a 1.65). Los modelos de referencia sí se reprodujeron, lo que descarta diferencias en
  datos y métricas.
- **RQ2 (series cortas):** el pipeline con torneo supera al naive estacional en la mayoría de los productos
  desde 12 meses de historia, pero no supera al naive simple en WAPE agregado y genera más sobrepronósticos
  extremos; la reparación de quiebres corrige la historia, sin efecto significativo en el error de
  pronóstico en este diseño.
- Estos resultados limitan las afirmaciones de superioridad del FTGM que pueden hacerse en la tesis y
  aportan una línea de trabajo futuro: estabilizar el FTGM semanal (T = 52 con menos de dos temporadas) y
  revisar el tope de divergencia de 50 veces el máximo histórico.

## Reproducción
Ver `README.md`. Orden: `exp1_replicacion_ye.py`, `exp2_diagnostico_gris.py`, `exp3_series_cortas.py`,
`exp4_pruebas.py`, `figuras.py`, `verificaciones.py`. Tiempo total de reloj: aproximadamente 9 min 20 s (7:29 + 0:14 + 1:13 +
0:23, más las figuras).
