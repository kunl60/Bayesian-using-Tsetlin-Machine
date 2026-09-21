"""
DAGMA run on Parkinsons Telemonitoring (OpenML), a real-world speech/voice-
signal regression dataset with an unknown ground-truth DAG. Discovers the
skeleton/Markov Blankets via DAGMA (a log-det acyclicity reformulation of
NOTEARS for continuous linear-Gaussian data), then scores each node's
discovered MB (and, for comparison, all raw features) as predictors for that
node via Linear Regression / KNN / SVR (see
experiments_real_regression/_common.py).
"""

from data.generate_regression import load_parkinsons_telemonitoring_dataset
from algorithms.dagma import DagmaAlgorithm
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    # 20 total columns (16 voice-signal measures, age/test_time, 2 UPDRS
    # targets) -> row-subsampled by the loader's sample_size=3000 default
    # (see data/generate_regression.py).
    raw_samples, _ = load_parkinsons_telemonitoring_dataset()

    # Continuous linear-Gaussian data -> DAGMA's log-det acyclicity score.
    algo = DagmaAlgorithm()
    discovery_results = algo.run(raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== DAGMA Results (Parkinsons Telemonitoring) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
