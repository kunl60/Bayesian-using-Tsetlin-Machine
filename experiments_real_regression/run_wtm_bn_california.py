from data.generate_regression import load_california_housing_dataset
from algorithms.wtm_bn_regressor import TsetlinMBElbowContinuousRegressor
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    raw_samples, categorical_samples = load_california_housing_dataset()

    # T sets the regressor's output resolution (unlike the classifier's T,
    # which is just a voting threshold) - needs to be much larger than a
    # classifier's T, along with more clauses/epochs, mirroring the values
    # validated for ECOLI70 (see run_ecoli_wtm_bn_regressor.py).
    algo = TsetlinMBElbowContinuousRegressor(
        num_epochs=5, number_clauses=10, top_n=5, T=10, s=1
    )
    discovery_results = algo.run(categorical_samples, raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== Tsetlin MB Elbow Results (California Housing, TMRegressor + Fisher's Z) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
