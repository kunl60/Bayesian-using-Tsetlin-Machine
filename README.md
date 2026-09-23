# Bayesian-using-Tsetlin-Machine

Markov Blanket discovery and Bayesian network structure learning with Tsetlin Machines.

A Tsetlin Machine (TM) is trained to predict each variable from all the others. The literals that appear in its learned clauses, weighted by clause weight, are used to shortlist candidate neighbours for that variable. Standard conditional-independence (CI) tests then confirm or reject each candidate, so the expensive CI search runs only on the TM shortlist and not on every variable pair. The resulting skeleton is oriented into a CPDAG using collider detection and Meek's rules.

The repository contains the Tsetlin Machine methods, for both discrete and continuous data, together with baselines and the scripts used to compare them.

## Methods

| Name | File | Data | Candidate shortlist |
|---|---|---|---|
| **WTM-NL** | [algorithms/wtm_nl.py](algorithms/wtm_nl.py) (`TsetlinMB`) | discrete | fixed `top_n` per target |
| **WTM-BN** | [algorithms/wtm_bn.py](algorithms/wtm_bn.py) (`TsetlinMB_elbow`) | discrete | relative cutoff on the TM scores, with `top_n` as a minimum |
| **WTM-NL regressor** | [algorithms/wtm_nl_regressor.py](algorithms/wtm_nl_regressor.py) | continuous | fixed `top_n` per target |
| **WTM-BN regressor** | [algorithms/wtm_bn_regressor.py](algorithms/wtm_bn_regressor.py) | continuous | relative cutoff on the TM scores |

The regressor variants replace the TM classifier with a TM regressor and use Fisher's Z as the CI test.

### Baselines

| Baseline | File |
|---|---|
| BOSS (Best Order Score Search) | [algorithms/boss.py](algorithms/boss.py) |
| CPC | [algorithms/cpc.py](algorithms/cpc.py), vendored code in [algorithms/cpc_lib/](algorithms/cpc_lib/) |
| DAGMA | [algorithms/dagma.py](algorithms/dagma.py) |
| XGES | [algorithms/xges.py](algorithms/xges.py) |
| Mutual-information Markov Blanket with relative-cutoff selection | [algorithms/mi_bn.py](algorithms/mi_bn.py), [algorithms/mi_bn_regressor.py](algorithms/mi_bn_regressor.py) |

## How the Tsetlin Machine methods work

1. **Propose.** Train one TM per variable, with that variable's own columns removed from the inputs. Score each other variable by (number of clauses it appears in) × (sum of those clauses' weights) and keep the strongest as that variable's candidates. WTM-NL keeps a fixed `top_n`. WTM-BN uses a relative cutoff instead: it cuts the sorted scores at the largest proportional drop between consecutive scores, keeping at least `top_n`.
2. **Skeleton.** For each variable, run a shrinking parents-and-children search over its candidates. Candidates are tested weakest first, and conditioning sets of size 1 up to `maxK` are drawn from the strongest remaining candidates. A candidate that some set separates from the target is removed, and the separating set is recorded. The union of all surviving edges is the skeleton.
3. **Colliders.** For each unshielded triple X – Y – T, find a separating set for X and T, then test whether conditioning on Y makes them dependent. If it does, `X → Y ← T` is a v-structure, and X and T are spouses.
4. **Orientation.** Orient the v-structures, then propagate with Meek's rules R1–R3 (never creating a cycle) to obtain a CPDAG. A variable's Markov Blanket is its neighbours in the skeleton plus its spouses.

`run()` returns a dictionary with the skeleton (`model`), the `cpdag`, `colliders`, `spouses`, `separating_sets`, the CI-test count (`ci_tests`) and the `runtime`.

## Repository layout

```
algorithms/                 TM methods and baselines
evaluation/metrics.py       graph metrics (SHD, precision/recall/F1, orientation accuracy, ...)
data/                       dataset loaders (generate_*.py) and cached CSVs
experiments/                ECOLI70 (continuous, known ground truth)
experiments_categorical/    discrete benchmark network (known ground truth)
experiments_real_categorical/  real categorical datasets (no ground truth)
experiments_real_regression/   real continuous datasets (no ground truth)
plots/                      result figures
```

## Setup

The project is developed with [pixi](https://pixi.sh) (`pixi.toml`, currently `osx-arm64` only). A plain `pip` list is in `requirement.txt`.

```bash
pixi install
pixi shell
```

Main dependencies: `tmu`, `pgmpy`, `causal-learn`, `networkx`, `scikit-learn`, `pandas`, `numpy`, `dagma`, `xges`, `gcastle`, `pulp`.

**`tmu` note.** The Tsetlin Machine implementation comes from [cair/tmu](https://github.com/cair/tmu). The experiments were run against a local checkout in which the x86-only `-mrdrnd` compiler flag was removed from `[tool.cffi_builder]` in its `pyproject.toml`, so that it builds on Apple Silicon.

## Running experiments

Run everything from the repository root, as modules. Every script prints its metrics to the terminal.

```bash
# discrete benchmark network, with ground truth (edit the network in data/generate_sachs.py)
python -m experiments_categorical.run_wtm_nl
python -m experiments_categorical.run_wtm_bn
python -m experiments_categorical.run_boss

# real categorical datasets
python -m experiments_real_categorical.run_wtm_nl_mushroom
python -m experiments_real_categorical.run_wtm_bn_spect

# ECOLI70, continuous with ground truth
python -m experiments.run_ecoli_wtm_nl_regressor
python -m experiments.run_ecoli_summary          # all methods side by side

# real regression datasets
python -m experiments_real_regression.run_wtm_nl_california
python -m experiments_real_regression.run_wtm_bn_superconductivity
```

File names follow `run_<method>[_<dataset>].py`, where `<method>` is one of `wtm_nl`, `wtm_bn`, `boss`, `cpc`, `mi_bn`, `dagma`, `xges`.

## Datasets

| Setting | Data | Loader |
|---|---|---|
| Discrete, known graph | pgmpy example network, sampled with `BayesianModelSampling` | [data/generate_sachs.py](data/generate_sachs.py) |
| Continuous, known graph | ECOLI70 (linear-Gaussian) | [data/generate_ecoli.py](data/generate_ecoli.py) |
| Real categorical | SPECT, Neticusdroid, Mushroom, TUANDROMD | [data/generate_real_categorical.py](data/generate_real_categorical.py) |
| Real continuous | California Housing, Superconductivity, Musk, ECG5000, Gas Turbine Emission, plus Parkinsons Telemonitoring (downloaded from OpenML on first use) | [data/generate_regression.py](data/generate_regression.py) |

Continuous data is binned or thermometer-encoded for the TM front end, while the CI tests use the raw values. Some datasets are cached under `data/` after the first download.

## Evaluation

- **Known ground truth** (discrete benchmark, ECOLI70): structural Hamming distance, skeleton precision/recall/F1, directed precision/recall/F1 and orientation accuracy on the CPDAG, CI tests per correct edge, and runtime. All are averaged over several seeds.
- **No ground truth** (real datasets): each variable is predicted from its discovered Markov Blanket. Categorical datasets use SVC, Naive Bayes and KNN and report accuracy. Continuous datasets use linear regression, KNN and SVR and report R². Splits are fixed by seed.

The metric functions are in [evaluation/metrics.py](evaluation/metrics.py).
