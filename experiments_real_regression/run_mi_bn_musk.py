from data.generate_regression import load_musk_dataset
from algorithms.mi_bn_regressor import MutualInfoMBElbowContinuousRegressor
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    # 167 features -> row-subsampled by the loader's sample_size=3000 default
    # (see data/generate_regression.py) to keep the CI-test-heavy PC search
    # tractable.
    raw_samples, categorical_samples = load_musk_dataset()

    # Mutual-information ablation of TsetlinMBElbowContinuousRegressor's
    # scoring step (see algorithms/mi_bn_regressor.py) - same
    # elbow cut, Fisher's Z CI testing, collider detection, and Meek's
    # rules, only mutual_info_regression(V_j, Y) replaces the TMRegressor
    # clause frequency x weight score.
    algo = MutualInfoMBElbowContinuousRegressor(top_n=5)
    discovery_results = algo.run(categorical_samples, raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== MI MB Elbow Results (Musk, mutual_info_regression + Fisher's Z) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
