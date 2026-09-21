import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from algorithms.cpc import CPCAlgorithm
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from data.generate_real_categorical import load_spect_dataset

# 1. Setup Data - real-world categorical, uniform binary domain (like
# neticusdroid.csv/tuandromd.csv), but a much smaller dataset (267 rows).
# Whole dataset, no subsampling.
full_df = load_spect_dataset()

# Category counts per column (full dataset), so CategoricalNB doesn't error
# on a category that's missing from a given train/test split fold
n_categories = full_df.nunique().to_dict()

# 2. Discover Markov Blankets using CPC
cpc = CPCAlgorithm(tester="chisq", alpha=0.01, max_cond_vars=3)
discovery_results = cpc.run(full_df)

# Extract MB_results from the graph K
# In a Markov Blanket, a node's neighbors in the moralized graph are its MB
K = discovery_results["model"]

MB_results = {node: list(K.neighbors(node)) for node in K.nodes()}

# Each run uses a different but fixed split seed, so the 5 runs differ from
# each other but the whole script is reproducible across executions
run_svc_means = []
run_nb_means = []
run_knn_means = []
run_seeds = [1, 2, 3, 4, 5]

for run, seed in enumerate(run_seeds, start=1):
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=seed)

    svc_accuracies = []
    nb_accuracies = []
    knn_accuracies = []

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
