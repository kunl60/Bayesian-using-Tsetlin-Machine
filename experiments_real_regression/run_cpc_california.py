"""
CPC run on California Housing (OpenML), a real-world regression dataset
with an unknown ground-truth DAG. Discovers the skeleton/Markov Blankets via
CPC (a causal-learn based PC/FCI variant with kPC-style PAG orientation
rules), then scores each node's discovered MB (and, for comparison, all raw
features) as predictors for that node via Linear Regression / KNN / SVR (see
experiments_real_regression/_common.py).
"""

from data.generate_regression import load_california_housing_dataset
from algorithms.cpc import CPCAlgorithm
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    raw_samples, _ = load_california_housing_dataset()

    # Continuous data -> Fisher-Z CI test.
    # max_cond_vars=4 here (not 3) because CPC's own loop tests conditioning
    # sets of size 0..k-1, so k=4 is what actually reaches size-3 sets -
    # matching PC/TM-MB's max_cond_vars=3 (0..3 inclusive).
    algo = CPCAlgorithm(tester="fisherz", alpha=0.01, max_cond_vars=3)
    discovery_results = algo.run(raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== CPC Results (Fisher-Z, California Housing) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
