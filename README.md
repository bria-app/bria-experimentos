# bria-experimentos

Scripts, raw results and execution logs behind the technical evaluation of the paper
*A Decision Support System for Inventory Replenishment in Retail Micro and Small Enterprises:
Reproducing the Fourier Time-Varying Grey Model and Evaluating Decisions on Short Series*
(Cruz Ticona, Ayquipa Ubaldo and Mori Yzaguirre, Universidad Peruana de Ciencias Aplicadas).

The system itself (web, API and forecasting engine) is published in the
[bria-app](https://github.com/bria-app) organization.

## Experiments in the paper

| Paper | Script | Main outputs |
|---|---|---|
| Exp. 1: fidelity replication of Ye et al. (2024) | `exp1_replicacion_ye.py` | `results/exp1_*.csv`, `logs/exp1_log.txt` |
| Fidelity investigation (50 variants) | `fidelidad/fid_run.py`, `fidelidad/consolidar.py` | `fidelidad/results/`, `fidelidad/FIDELIDAD.md` |
| Exp. 2: grey accuracy grades | `exp2_diagnostico_gris.py` | `results/exp2_*.csv` |
| Exp. 3 and 3b: 300 short series, injected censoring | `exp3_series_cortas.py` | `results/exp3_*.csv` |
| Holm adjustment of Exps. 1, 3 and 3b | `exp3_holm.py` | `results/*_holm.csv` |
| Exp. 4: decision simulation | `exp7_simulacion_decision.py` | `results/exp7_*.csv`, `logs/exp7_log.txt` |
| Exp. 4: smoothed references (last month, 4-week and 3-month averages) | computed from `results/exp7b_referencias_extra_por_producto.csv` | `results/exp7b_*.csv` |
| Engine tests and latency | `exp4_pruebas.py` | `results/exp4_latencia.csv`, `logs/exp4_*.txt` |
| Engine regression after the audit (397 cases) | `regresion_motor_postfix.py` | `logs/engine_postfix_*.txt` |
| Paper figures (600 dpi) | `make_paper_figures.py` | `paper_figures/` |

The Exp. 4 of the paper is called Exp. 7 in the scripts, because the thesis numbers its experiments
differently. `RESULTADOS.md` (in Spanish) documents every protocol, number and limitation.

## Engine revision

All experiments ran against commit
[`be6a98f`](https://github.com/Inventory-DSS-Legacy/inventory-dss-ftgm-engine/commit/be6a98f8bb3a6ebc2c00f46a0ef7a0dc0c28e456)
of the forecasting engine, with a clean working tree. The repository also contains the M5 files used
(`docs/ftgmmodel/sales_data1.csv` and `sales_data2.csv`, days d_1 to d_1941, as distributed with the
original FTGM code).

## Reproducing

```bash
git clone https://github.com/Inventory-DSS-Legacy/inventory-dss-ftgm-engine.git engine
git -C engine checkout be6a98f8bb3a6ebc2c00f46a0ef7a0dc0c28e456
pip install -e ./engine pandas statsmodels scikit-learn matplotlib
export ENGINE_DIR=$PWD/engine ENGINE_COMMIT=be6a98f8bb3a6ebc2c00f46a0ef7a0dc0c28e456
python exp1_replicacion_ye.py
python exp2_diagnostico_gris.py
python exp3_series_cortas.py
python exp7_simulacion_decision.py
python exp3_holm.py
python make_paper_figures.py
```

The first run builds `cache/` from the M5 files. Exp. 3 takes about 1 minute and Exp. 4 about
3 minutes with 16 processes. The original environment was Python 3.11.4 on Windows 11 with
numpy 2.4.4, scipy 1.16.3, pandas 2.3.3, statsmodels 0.14.6, scikit-learn 1.7.2 and matplotlib 3.10.7.

## Data

M5 is the dataset of the M5 forecasting competition (Makridakis, Spiliotis and Assimakopoulos, 2022).
It is not micro-enterprise data: it is used because it is the dataset of the original FTGM study and
allows controlled experiments. Stockouts in Exp. 3b are injected synthetically. No data from the
user sessions is included.
