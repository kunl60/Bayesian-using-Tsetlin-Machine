"""
DAGMA run on California Housing (OpenML), a real-world regression dataset
with an unknown ground-truth DAG. Discovers the skeleton/Markov Blankets via
DAGMA (a log-det acyclicity reformulation of NOTEARS for continuous linear-
Gaussian data), then scores each node's discovered MB (and, for comparison,
all raw features) as predictors for that node via Linear Regression / KNN /
SVR (see experiments_real_regression/_common.py).
"""

from data.generate_regression import load_california_housing_dataset
from algorithms.dagma import DagmaAlgorithm
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    raw_samples, _ = load_california_housing_dataset()

    # Continuous linear-Gaussian data -> DAGMA's log-det acyclicity score.
    algo = DagmaAlgorithm()
    discovery_results = algo.run(raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== DAGMA Results (California Housing) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
