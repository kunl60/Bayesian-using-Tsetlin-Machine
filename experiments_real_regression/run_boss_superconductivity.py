"""
BOSS run on Superconductivity (OpenML), a real-world regression dataset
with an unknown ground-truth DAG. Discovers the skeleton/Markov Blankets,
then scores each node's discovered MB (and, for comparison, all raw
features) as predictors for that node via Linear Regression / KNN / SVR
(see experiments_real_regression/_common.py).
"""

from data.generate_regression import load_superconductivity_dataset
from algorithms.boss import BOSSAlgorithm
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    # 81 features -> row-subsampled by the loader's sample_size=3000 default
    # (see data/generate_regression.py) to keep this tractable.
    raw_samples, _ = load_superconductivity_dataset()

    # Continuous data -> BIC (covariance-based) score.
    algo = BOSSAlgorithm(score_func="local_score_BIC_from_cov")
    discovery_results = algo.run(raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== BOSS Results (BIC, Superconductivity) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
