import numpy as np
from data.generate_ecoli import load_ecoli_dataset
from algorithms.cpc import CPCAlgorithm
from evaluation.metrics import (
    structural_hamming_distance,
    precision_recall_f1,
    get_nodes_difference,
    ci_tests_per_correct_edge,
    directed_precision_recall_f1,
    orientation_accuracy,
)


def main():
    run_seeds = [1, 2, 3]

    shd_list, precision_list, recall_list, f1_list = [], [], [], []
    ci_tests_list, ci_per_edge_list, runtime_list, nodes_diff_list = [], [], [], []
    d_precision_list, d_recall_list, d_f1_list, orient_acc_list = [], [], [], []

    for run, seed in enumerate(run_seeds, start=1):
        raw_samples, ground_truth, _, _ = load_ecoli_dataset(seed=seed)

        # Continuous data -> Fisher-Z CI test.
        cpc = CPCAlgorithm(tester="fisherz", alpha=0.01, max_cond_vars=3)
        results = cpc.run(raw_samples)

        shd, fp, fn = structural_hamming_distance(ground_truth, results["model"])
        precision, recall, f1 = precision_recall_f1(ground_truth, results["model"])
        nodes_diff = get_nodes_difference(ground_truth, results["model"])
        ci_per_edge = ci_tests_per_correct_edge(results["ci_tests"], ground_truth, results["model"])

        d_precision, d_recall, d_f1 = directed_precision_recall_f1(ground_truth, results["cpdag"])
        correct, incorrect, undirected, orient_acc = orientation_accuracy(ground_truth, results["cpdag"])

        print(f"\n--- Run {run} (seed={seed}) ---")
        print(f"CI Tests: {results['ci_tests']}")
        print(f"CI Tests per Correct Edge: {ci_per_edge:.2f}")
        print(f"Runtime (s): {results['runtime']:.2f}")
        print(f"Nodes Difference: {nodes_diff}")
        print(f"SHD: {shd}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Directed Precision: {d_precision:.4f}")
        print(f"Directed Recall: {d_recall:.4f}")
        print(f"Directed F1 Score: {d_f1:.4f}")
        print(f"Correctly Oriented Edges: {correct}")
        print(f"Incorrectly Oriented Edges: {incorrect}")
        print(f"Left Undirected: {undirected}")
        print(f"Orientation Accuracy (resolved only): {orient_acc:.4f}")

        shd_list.append(shd)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        ci_tests_list.append(results["ci_tests"])
        ci_per_edge_list.append(ci_per_edge)
        runtime_list.append(results["runtime"])
        nodes_diff_list.append(nodes_diff)
        d_precision_list.append(d_precision)
        d_recall_list.append(d_recall)
        d_f1_list.append(d_f1)
        orient_acc_list.append(orient_acc)

    print("\n===== CPC Results (Fisher-Z, ECOLI70) - Summary (mean ± std across runs) =====")
    print(f"CI Tests: {np.mean(ci_tests_list):.2f} ± {np.std(ci_tests_list):.2f}")
    print(f"CI Tests per Correct Edge: {np.mean(ci_per_edge_list):.2f} ± {np.std(ci_per_edge_list):.2f}")
    print(f"Runtime (s): {np.mean(runtime_list):.2f} ± {np.std(runtime_list):.2f}")
    print(f"Nodes Difference: {np.mean(nodes_diff_list):.2f} ± {np.std(nodes_diff_list):.2f}")
    print(f"SHD: {np.mean(shd_list):.2f} ± {np.std(shd_list):.2f}")
    print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1 Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    print("\n--- Direction (PAG) - Summary (mean ± std across runs) ---")
    print(f"Directed Precision: {np.mean(d_precision_list):.4f} ± {np.std(d_precision_list):.4f}")
    print(f"Directed Recall: {np.mean(d_recall_list):.4f} ± {np.std(d_recall_list):.4f}")
    print(f"Directed F1 Score: {np.mean(d_f1_list):.4f} ± {np.std(d_f1_list):.4f}")
    print(f"Orientation Accuracy (resolved only): {np.mean(orient_acc_list):.4f} ± {np.std(orient_acc_list):.4f}")


if __name__ == "__main__":
    main()
