"""
CPC run on Parkinsons Telemonitoring (OpenML), a real-world speech/voice-signal
regression dataset with an unknown ground-truth DAG. Discovers the
skeleton/Markov Blankets via CPC (a causal-learn based PC/FCI variant with kPC-
style PAG orientation rules), then scores each node's discovered MB (and, for
comparison, all raw features) as predictors for that node via Linear Regression
/ KNN / SVR (see experiments_real_regression/_common.py).
"""

from data.generate_regression import load_parkinsons_telemonitoring_dataset
from algorithms.cpc import CPCAlgorithm
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

    # Continuous data -> Fisher-Z CI test.
    algo = CPCAlgorithm(tester="fisherz", alpha=0.01, max_cond_vars=3)
    discovery_results = algo.run(raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== CPC Results (Fisher-Z, Parkinsons Telemonitoring) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
