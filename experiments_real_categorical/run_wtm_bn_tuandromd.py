import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from algorithms.wtm_bn import TsetlinMB_elbow
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from data.generate_real_categorical import load_tuandromd_dataset

# 1. Setup Data - row-subsampled to 3000 (was the full 4465 rows, unbounded)
full_df = load_tuandromd_dataset(sample_size=3000, seed=42)

# Category counts per column (full dataset), so CategoricalNB doesn't error
# on a category that's missing from a given train/test split fold
n_categories = full_df.nunique().to_dict()

# 2. Discover Markov Blankets using the elbow-cut Tsetlin Machine variant
# (candidate count per target picked via find_elbow_index instead of a
# fixed top_n) - see algorithms/wtm_bn.py.
tm_mb_elbow = TsetlinMB_elbow(num_epochs=5, number_clauses=10, top_n=5, T=10, s=5)
discovery_results = tm_mb_elbow.run(full_df, full_df.astype(str))

# Extract MB_results from the graph K
# In a Markov Blanket, a node's neighbors in the moralized graph are its MB
K = discovery_results["model"]

MB_results = {node: list(K.neighbors(node)) for node in K.nodes()}

# Elbow-selected variables per target (deduplicated variable names, i.e.
# not raw pos/neg TM literals) - counted before CI-test pruning. Discovery
# runs once (unlike the classifier eval below), so this is a single set
# of stats, not mean ± std across runs.
elbow_counts = {target: len(vars_) for target, vars_ in discovery_results["elbow_selected"].items()}
elbow_values = list(elbow_counts.values())
elbow_mean = np.mean(elbow_values)
elbow_median = np.median(elbow_values)
elbow_max = np.max(elbow_values)

# 3. Evaluate over multiple runs, each with a different but fixed split seed
# so the 5 splits differ from each other but the whole script is reproducible
run_seeds = [1, 2, 3, 4, 5]
run_svc_means = []
run_nb_means = []
run_knn_means = []

for run, seed in enumerate(run_seeds, start=1):
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=seed)

    svc_accuracies = []
    nb_accuracies = []
    knn_accuracies = []

    # Evaluate Features with Multiple Classifiers
    for label_col, feature_cols in MB_results.items():
        if not feature_cols:
            svc_accuracies.append(0)
            nb_accuracies.append(0)
            knn_accuracies.append(0)
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[label_col]

        X_test = test_df[feature_cols]
        y_test = test_df[label_col]

        # Skip targets with only a single class in this split's training data -
        # SVC/CategoricalNB/KNN can't fit or score a one-class target.
        if y_train.nunique() < 2:
            svc_accuracies.append(0)
            nb_accuracies.append(0)
            knn_accuracies.append(0)
            continue

        # ---- SVC ----
        svc = SVC(kernel="rbf", gamma="scale")
        svc.fit(X_train, y_train)
        svc_pred = svc.predict(X_test)
        svc_accuracies.append(accuracy_score(y_test, svc_pred))

        # ---- Naive Bayes ----
        nb = CategoricalNB(min_categories=[n_categories[c] for c in feature_cols])
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)
        nb_accuracies.append(accuracy_score(y_test, nb_pred))

        # ---- KNN ----
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accuracies.append(accuracy_score(y_test, knn_pred))

    print(f"\n--- Run {run} (seed={seed}) ---")
    print(f"SVC Mean Accuracy: {100 * np.mean(svc_accuracies):.2f}%")
    print(f"Naive Bayes Mean Accuracy: {100 * np.mean(nb_accuracies):.2f}%")
    print(f"KNN Mean Accuracy: {100 * np.mean(knn_accuracies):.2f}%")

    run_svc_means.append(np.mean(svc_accuracies))
    run_nb_means.append(np.mean(nb_accuracies))
    run_knn_means.append(np.mean(knn_accuracies))

# 5. Final Summary
print("\n--- Summary ---")
print(f"CI Tests: {discovery_results['ci_tests']}")
print(f"Runtime (s): {discovery_results['runtime']:.2f}")
print(f"Elbow-Selected Variables per Target: {elbow_counts}")
print(f"Elbow-Selected Variables - Mean: {elbow_mean:.2f}")
print(f"Elbow-Selected Variables - Median across targets: {elbow_median:.2f}")
print(f"Elbow-Selected Variables - Max across targets: {elbow_max}")

print("\n--- Accuracy Results (mean ± std across runs) ---")
print(f"SVC Mean Accuracy: {100 * np.mean(run_svc_means):.2f}% ± {100 * np.std(run_svc_means):.2f}%")
print(f"Naive Bayes Mean Accuracy: {100 * np.mean(run_nb_means):.2f}% ± {100 * np.std(run_nb_means):.2f}%")
print(f"KNN Mean Accuracy: {100 * np.mean(run_knn_means):.2f}% ± {100 * np.std(run_knn_means):.2f}%")

# MB Size Metrics
mb_sizes = [len(members) for members in MB_results.values()]
num_nodes = len(MB_results)
average_mb_size = np.mean(mb_sizes) if num_nodes > 0 else 0
std_mb = np.std(mb_sizes)

print("\n--- Markov Blanket Stats ---")
print(f"Average MB size per node: {average_mb_size:.2f}")
print(f"Std MB size per node: {std_mb:.2f}")
