"""
Shared evaluation helpers for the experiments_real_regression/ scripts.

These mirror the per-script classifier-accuracy evaluation used in
experiments_real_categorical/ (fit a model on each discovered Markov
Blanket, score it), but with regressors/R2 instead of classifiers/accuracy,
since every dataset here (California Housing, Superconductivity) has a
continuous target and no known ground-truth DAG - so there's no
precision/recall to report, only how predictive each discovered MB is.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RUN_SEEDS = [1, 2, 3, 4, 5]


def _make_svr():
    # SVR's epsilon=0.1 is an absolute tolerance band, not relative to the
    # target's scale - for a small-absolute-range continuous column (e.g.
    # a jitter/shimmer-style ratio spanning ~0.001-0.17), that band covers
    # almost the entire signal, so unscaled SVR collapses to a near-constant
    # prediction and R2 (which divides by the target's own tiny variance)
    # blows up to something like -9700%. TransformedTargetRegressor scales
    # y the same way the outer Pipeline already scales X, then inverse-
    # transforms predictions before scoring - so R2 is still computed in the
    # original target units and stays comparable to Linear Regression/KNN.
    return TransformedTargetRegressor(
        regressor=SVR(kernel="rbf"), transformer=StandardScaler()
    )


def discover_markov_blankets(discovery_results):
    """
    discovery_results['model'] -> {node: [neighbors...]}, the skeleton's
    PC (parents/children) set. Prediction here uses PC only (matching
    experiments_real_categorical) - spouses are intentionally not merged
    in, even when discovery_results['spouses'] is available.
    """
    model = discovery_results["model"]
    return {node: list(model.neighbors(node)) for node in model.nodes()}


def print_discovery_summary(discovery_results, mb_results):
    print("\n--- Summary ---")
    print(f"CI Tests: {discovery_results.get('ci_tests')}")
    print(f"Runtime (s): {discovery_results['runtime']:.2f}")

    mb_sizes = [len(members) for members in mb_results.values()]
    num_nodes = len(mb_results)
    average_mb_size = np.mean(mb_sizes) if num_nodes > 0 else 0
    std_mb = np.std(mb_sizes)

    print("\n--- Markov Blanket Stats ---")
    print(f"Average MB size per node: {average_mb_size:.2f}")
    print(f"Std MB size per node: {std_mb:.2f}")


def evaluate_markov_blankets(raw_samples, mb_results, run_seeds=RUN_SEEDS):
    """
    For each node, predict it from its discovered MB (a node with an empty
    MB scores 0.0) using three regressors, over multiple fixed train/test
    splits, then print per-run and mean +/- std R2 plus MB size stats.
    """
    run_means = {"linreg": [], "knn": [], "svr": []}

    for run, seed in enumerate(run_seeds, start=1):
        train_df, test_df = train_test_split(
            raw_samples, test_size=0.20, random_state=seed
        )

        linreg_r2 = []
        knn_r2 = []
        svr_r2 = []

        for label_col, feature_cols in mb_results.items():
            if not feature_cols:
                linreg_r2.append(0.0)
                knn_r2.append(0.0)
                svr_r2.append(0.0)
                continue

            X_train = train_df[feature_cols]
            y_train = train_df[label_col]
            X_test = test_df[feature_cols]
            y_test = test_df[label_col]

            linreg = LinearRegression()
            linreg.fit(X_train, y_train)
            linreg_r2.append(linreg.score(X_test, y_test))

            # KNN/SVR are distance/kernel-based - scale features first (fit
            # the scaler on train only, via Pipeline, so test stays unseen)
            # or a large-magnitude column like totalrooms swamps a
            # small-magnitude one like latitude in the distance metric.
            knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
            knn.fit(X_train, y_train)
            knn_r2.append(knn.score(X_test, y_test))

            svr = make_pipeline(StandardScaler(), _make_svr())
            svr.fit(X_train, y_train)
            svr_r2.append(svr.score(X_test, y_test))

        print(f"\n--- Run {run} (seed={seed}) ---")
        print(f"Linear Regression Mean R2: {np.mean(linreg_r2) * 100:.2f}%")
        print(f"KNN Regressor Mean R2: {np.mean(knn_r2) * 100:.2f}%")
        print(f"SVR Mean R2: {np.mean(svr_r2) * 100:.2f}%")

        run_means["linreg"].append(np.mean(linreg_r2))
        run_means["knn"].append(np.mean(knn_r2))
        run_means["svr"].append(np.mean(svr_r2))

    print("\n--- R2 Results (mean ± std across runs) ---")
    for name, label in zip(
        ("linreg", "knn", "svr"), ("Linear Regression", "KNN Regressor", "SVR")
    ):
        vals = run_means[name]
        print(f"{label} Mean R2: {np.mean(vals) * 100:.2f}% ± {np.std(vals) * 100:.2f}%")

    return run_means


def evaluate_all_features(raw_samples, target_cols=None, run_seeds=RUN_SEEDS):
    """
    Baseline for comparison against evaluate_markov_blankets: for each target
    column, predict it from *every other* raw column instead of just its
    discovered MB, using the same three regressors over the same fixed
    train/test splits. Shows how much (if any) predictive power the MB
    selection is leaving on the table versus using all available features.
    """
    if target_cols is None:
        target_cols = raw_samples.columns.tolist()

    all_cols = raw_samples.columns.tolist()

    run_means = {"linreg": [], "knn": [], "svr": []}

    for run, seed in enumerate(run_seeds, start=1):
        train_df, test_df = train_test_split(
            raw_samples, test_size=0.20, random_state=seed
        )

        linreg_r2 = []
        knn_r2 = []
        svr_r2 = []

        for label_col in target_cols:
            feature_cols = [c for c in all_cols if c != label_col]

            X_train = train_df[feature_cols]
            y_train = train_df[label_col]
            X_test = test_df[feature_cols]
            y_test = test_df[label_col]

            linreg = LinearRegression()
            linreg.fit(X_train, y_train)
            linreg_r2.append(linreg.score(X_test, y_test))

            # KNN/SVR are distance/kernel-based - scale features first (fit
            # the scaler on train only, via Pipeline, so test stays unseen)
            # or a large-magnitude column like totalrooms swamps a
            # small-magnitude one like latitude in the distance metric.
            knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
            knn.fit(X_train, y_train)
            knn_r2.append(knn.score(X_test, y_test))

            svr = make_pipeline(StandardScaler(), _make_svr())
            svr.fit(X_train, y_train)
            svr_r2.append(svr.score(X_test, y_test))

        print(f"\n--- All Features: Run {run} (seed={seed}) ---")
        print(f"Linear Regression Mean R2: {np.mean(linreg_r2) * 100:.2f}%")
        print(f"KNN Regressor Mean R2: {np.mean(knn_r2) * 100:.2f}%")
        print(f"SVR Mean R2: {np.mean(svr_r2) * 100:.2f}%")

        run_means["linreg"].append(np.mean(linreg_r2))
        run_means["knn"].append(np.mean(knn_r2))
        run_means["svr"].append(np.mean(svr_r2))

    print("\n--- All Features: R2 Results (mean ± std across runs) ---")
    for name, label in zip(
        ("linreg", "knn", "svr"), ("Linear Regression", "KNN Regressor", "SVR")
    ):
        vals = run_means[name]
        print(f"{label} Mean R2: {np.mean(vals) * 100:.2f}% ± {np.std(vals) * 100:.2f}%")

    return run_means
