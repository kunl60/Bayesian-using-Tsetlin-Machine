"""
Runs a set of structure-learning algorithms on ECOLI70 several times each and
reports mean ± std (across runs) for skeleton precision/recall/F1,
directed (CPDAG) precision/recall/F1, and runtime (s).

Each run draws a *fresh* ECOLI70 sample (a different `seed` passed to
load_ecoli_dataset) so the std reflects sampling variability, not just
algorithm-internal randomness. All algorithms within a given run see the
same sampled dataset, so the comparison stays apples-to-apples per run.

Algorithms: DAGMA, BOSS, CPC, Elbow Con Regressor (Tsetlin MB
Elbow Continuous Regressor), TM MB Con Regressor (Tsetlin MB Continuous
Regressor).

Usage:
    python -m experiments.run_ecoli_summary [--runs N] [--seed-base S] [--csv PATH]
"""

import argparse
import csv
import statistics

from data.generate_ecoli import load_ecoli_dataset
from algorithms.dagma import DagmaAlgorithm
from algorithms.boss import BOSSAlgorithm
from algorithms.cpc import CPCAlgorithm
from algorithms.wtm_bn_regressor import TsetlinMBElbowContinuousRegressor
from algorithms.wtm_nl_regressor import TsetlinMBContinuousRegressor
from evaluation.metrics import precision_recall_f1, directed_precision_recall_f1

METRIC_KEYS = ["precision", "recall", "f1", "d_precision", "d_recall", "d_f1", "runtime"]
# (display label, decimal places, is_percent) per metric, for the printed table header.
METRIC_DISPLAY = {
    "precision": ("Precision", 2, True),
    "recall": ("Recall", 2, True),
    "f1": ("F1", 2, True),
    "d_precision": ("Dir.Precision", 2, True),
    "d_recall": ("Dir.Recall", 2, True),
    "d_f1": ("Dir.F1", 2, True),
    "runtime": ("Runtime (s)", 2, False),
}


def _run_dagma(raw_samples, categorical_samples):
    dagma = DagmaAlgorithm(loss_type="l2", lambda1=0.03, w_threshold=0.3)
    results = dagma.run(raw_samples)
    return results["model"], results["cpdag"], results["runtime"]


def _run_boss(raw_samples, categorical_samples):
    boss = BOSSAlgorithm(score_func="local_score_BIC_from_cov")
    results = boss.run(raw_samples)
    return results["model"], results["cpdag"], results["runtime"]


def _run_cpc(raw_samples, categorical_samples):
    cpc = CPCAlgorithm(tester="fisherz", alpha=0.01, max_cond_vars=3)
    results = cpc.run(raw_samples)
    return results["model"], results["cpdag"], results["runtime"]


def _run_elbow_con_regressor(raw_samples, categorical_samples):
    tm = TsetlinMBElbowContinuousRegressor(
        num_epochs=1, number_clauses=100, top_n=5, T=100, s=1
    )
    results = tm.run(categorical_samples, raw_samples)
    return results["model"], results["cpdag"], results["runtime"]


def _run_tm_mb_con_regressor(raw_samples, categorical_samples):
    tm = TsetlinMBContinuousRegressor(num_epochs=1, number_clauses=100, top_n=2, T=100, s=1)
    results = tm.run(categorical_samples, raw_samples)
    return results["model"], results["cpdag"], results["runtime"]


ALGORITHMS = {
    "DAGMA": _run_dagma,
    "BOSS": _run_boss,
    "CPC": _run_cpc,
    "Elbow Con Regressor": _run_elbow_con_regressor,
    "TM MB Con Regressor": _run_tm_mb_con_regressor,
}


def evaluate_once(runner, ground_truth, raw_samples, categorical_samples):
    model, cpdag, runtime = runner(raw_samples, categorical_samples)
    precision, recall, f1 = precision_recall_f1(ground_truth, model)
    d_precision, d_recall, d_f1 = directed_precision_recall_f1(ground_truth, cpdag)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "d_precision": d_precision,
        "d_recall": d_recall,
        "d_f1": d_f1,
        "runtime": runtime,
    }


def summarize(runs):
    """runs: list of per-run metric dicts -> dict of key -> (mean, std)."""
    summary = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in runs]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[key] = (mean, std)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per algorithm")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="First run uses this seed for load_ecoli_dataset; each subsequent "
        "run increments it by 1, so a fresh ECOLI70 sample is drawn per run",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write summary CSV")
    args = parser.parse_args()

    all_runs = {name: [] for name in ALGORITHMS}
    for i in range(args.runs):
        seed = args.seed_base + i
        print(f"--- run {i + 1}/{args.runs} (seed={seed}) ---", flush=True)
        raw_samples, ground_truth, categorical_samples, _ = load_ecoli_dataset(seed=seed)

        for name, runner in ALGORITHMS.items():
            metrics = evaluate_once(runner, ground_truth, raw_samples, categorical_samples)
            all_runs[name].append(metrics)
            print(f"  {name}: {metrics}", flush=True)

    summaries = {name: summarize(runs) for name, runs in all_runs.items()}

    print(f"\n\n===== ECOLI70 SUMMARY (mean ± std across {args.runs} runs) =====")
    col_width = 18
    header = f"{'Algorithm':22s} | " + " | ".join(
        f"{METRIC_DISPLAY[k][0]:{col_width}s}" for k in METRIC_KEYS
    )
    print(header)
    print("-" * len(header))
    for name, summary in summaries.items():
        cells = []
        for k in METRIC_KEYS:
            mean, std = summary[k]
            _, decimals, is_percent = METRIC_DISPLAY[k]
            if is_percent:
                mean, std = mean * 100, std * 100
                suffix = "%"
            else:
                suffix = ""
            cells.append(f"{mean:.{decimals}f}{suffix} ± {std:.{decimals}f}{suffix}")
        print(f"{name:22s} | " + " | ".join(f"{c:{col_width}s}" for c in cells))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["algorithm"]
                + [f"{k}_mean" for k in METRIC_KEYS]
                + [f"{k}_std" for k in METRIC_KEYS]
            )
            for name, summary in summaries.items():
                means = [summary[k][0] for k in METRIC_KEYS]
                stds = [summary[k][1] for k in METRIC_KEYS]
                writer.writerow([name] + means + stds)
        print(f"\nWrote summary CSV to {args.csv}")


if __name__ == "__main__":
    main()
