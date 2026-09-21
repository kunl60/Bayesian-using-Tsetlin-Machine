from data.generate_regression import load_ecg5000_dataset
from algorithms.wtm_nl_regressor import TsetlinMBContinuousRegressor
from experiments_real_regression._common import (
    discover_markov_blankets,
    evaluate_markov_blankets,
    print_discovery_summary,
)


def main():
    # 141 features -> row-subsampled by the loader's sample_size=3000 default
    # (see data/generate_regression.py) to keep the per-column TM training
    # and CI-test-heavy PC search tractable.
    raw_samples, categorical_samples = load_ecg5000_dataset()

    # Same TM settings validated for ECOLI70/California - T sets the
    # regressor's output resolution (see run_wtm_nl_california.py).
    algo = TsetlinMBContinuousRegressor(
        num_epochs=5, number_clauses=50, top_n=5, T=50, s=1
    )
    discovery_results = algo.run(categorical_samples, raw_samples)

    mb_results = discover_markov_blankets(discovery_results)

    print("===== Tsetlin MB Results (ECG5000, TMRegressor + Fisher's Z) =====")
    print_discovery_summary(discovery_results, mb_results)
    evaluate_markov_blankets(raw_samples, mb_results)


if __name__ == "__main__":
    main()
