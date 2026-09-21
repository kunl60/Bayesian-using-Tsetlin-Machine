"""
XGES run on Gas Turbine CO and NOx Emission (UCI), a real-world industrial
sensor-signal regression dataset with an unknown ground-truth DAG. Discovers
the skeleton/Markov Blankets via XGES (Extremely Greedy Equivalence Search),
then scores each node's discovered MB (and, for comparison, all raw features)
as predictors for that node via Linear Regression / KNN / SVR (see
experiments_real_regression/_common.py).
"""

from data.generate_regression import load_gas_turbine_emission_dataset
from algorithms.xges import XGESAlgorithm
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    # 11 total columns (9 sensor readings + CO/NOx targets) ->
    # row-subsampled by the loader's sample_size=3000 default
    # (see data/generate_regression.py).
    raw_samples, _ = load_gas_turbine_emission_dataset()

    # Continuous data -> BIC (covariance-based) score.
    algo = XGESAlgorithm()
    discovery_results = algo.run(raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== XGES Results (BIC, Gas Turbine Emission) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
