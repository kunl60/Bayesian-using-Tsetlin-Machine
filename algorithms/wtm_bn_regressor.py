import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tmu.models.regression.vanilla_regressor import TMRegressor
import networkx as nx
from collections import defaultdict
import time
import itertools
from algorithms.ci_tests import fisher_z

class TsetlinMBElbowContinuousRegressor:

    def __init__(self, num_epochs, number_clauses, top_n, alpha=0.01, maxK=3, T=500, s=1):
        self.num_epochs = num_epochs
        self.number_clauses = number_clauses
        self.top_n = top_n
        self.alpha = alpha
        self.maxK = maxK
        self.T = T
        self.s = s
        self.ci_test_count = 0

    @staticmethod
    def clean_and_deduplicate(arr):
        cleaned = []
        for item in arr:
            cleaned.append(item.split('_')[0])

        cleaned = list(set(cleaned))
        return cleaned

    @staticmethod
    def find_elbow_index(scores):

        if len(scores) <= 5:
            return len(scores)

        # Compute differences between consecutive scores
        diffs = np.diff(scores)

        # Normalize by the preceding score so the cut reflects a
        # proportional collapse, not just the largest absolute gap
        # (which is almost always right after the top score)
        denom = np.array(scores[:-1], dtype=float)
        denom[denom == 0] = 1e-9
        relative_diffs = diffs / denom

        # Since scores are descending, diffs are negative
        # We want the largest relative drop → most negative value
        elbow_index = np.argmin(relative_diffs) + 1

        return elbow_index

    @staticmethod
    def calculate_input_output_frequency_weightage(cleaned_output, top_n):
        output_data = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "total_weightage": 0}))

        # Step 1: Aggregate
        for entry in cleaned_output:
            *inputs, output, weight = entry

            for input_element in inputs:
                output_data[output][input_element]["frequency"] += 1
                output_data[output][input_element]["total_weightage"] += weight

        result = {}
        selected_counts = {}

        # Step 2: Process each output
        for output, input_data in output_data.items():

            # Compute scores
            scored_inputs = [
                (input_element, values["frequency"] * values["total_weightage"])
                for input_element, values in input_data.items()
            ]

            # Sort descending (strongest association first)
            scored_inputs.sort(key=lambda x: x[1], reverse=True)

            # Extract scores only
            scores = [score for _, score in scored_inputs]

            # Step 3: Find elbow
            elbow_idx = TsetlinMBElbowContinuousRegressor.find_elbow_index(scores)

            # Step 4: Select up to elbow, but never fewer than top_n
            # (bounded by how many candidates actually exist)
            selected_count = max(elbow_idx, top_n)
            selected_count = min(selected_count, len(scored_inputs))
            selected_counts[output] = selected_count

            # Keep (input, score) pairs, still sorted descending - this
            # score doubles as the FSMB-style "dep" ranking downstream.
            result[output] = scored_inputs[:selected_count]

        print(f"selected_count per target: {selected_counts}")
        print(f"selected_count list: {list(selected_counts.values())}")

        return result

    def ci_test_counter(self, X, Y, Z, data):
        self.ci_test_count += 1
        return fisher_z(
            X=X, Y=Y, Z=Z, data=data,
            boolean=True,
            significance_level=self.alpha
        )

    @staticmethod
    def _sepset_key(a, b):
        return tuple(sorted((a, b)))

    def compute_target_PC(self, target, candidates_with_scores, raw_samples, separating_sets):

        score = dict(candidates_with_scores)

        # Zero-order gate, same role as the old per-pair pre-check: drop
        # anything TM selected that is in fact unconditionally independent
        # of the target before spending combinatorial tests on it.
        PC = []
        for X in score:
            if self.ci_test_counter(X, target, [], raw_samples):
                separating_sets[self._sepset_key(X, target)] = []
            else:
                PC.append(X)

        # FSMB ordering: test the weakest-scored survivor first, trying to
        # explain it away by conditioning on the strongest remaining
        # survivors first (mirrors FSMB's cpc_sorted / remaining_vars_sorted).
        for X in sorted(PC, key=lambda v: score[v]):
            if X not in PC:
                continue

            remaining_sorted = sorted(
                (v for v in PC if v != X), key=lambda v: score[v], reverse=True
            )

            max_size = min(self.maxK, len(remaining_sorted))
            removed = False

            for r in range(1, max_size + 1):
                for combo in itertools.combinations(remaining_sorted, r):
                    Z = list(combo)
                    if self.ci_test_counter(X, target, Z, raw_samples):
                        separating_sets[self._sepset_key(X, target)] = Z
                        removed = True
                        break
                if removed:
                    break

            if removed:
                PC.remove(X)

        return PC

    @staticmethod
    def _would_create_cycle(dag, u, v):
        """
        True if a directed path v -> ... -> u already exists using only
        edges that are already fully oriented (single direction). If so,
        adding u -> v would close a directed cycle.
        """
        oriented = nx.DiGraph(
            (p, q) for p, q in dag.edges() if not dag.has_edge(q, p)
        )
        if v not in oriented or u not in oriented:
            return False
        return nx.has_path(oriented, v, u)

    @staticmethod
    def _orient_edge(dag, u, v):
        """
        Orient u -> v in place. Left untouched (conflict) if the edge is
        already oriented the other way, or if orienting it this way would
        close a directed cycle - noisy/incomplete collider evidence can
        otherwise imply contradictory orientations, and the result must
        stay a valid DAG/CPDAG. Returns True if the graph changed (or was
        already oriented u -> v), False if left alone.
        """
        has_uv = dag.has_edge(u, v)
        has_vu = dag.has_edge(v, u)

        if not has_uv and not has_vu:
            return False  # not part of the skeleton

        if has_vu and not has_uv:
            return False  # already oriented the other way -> conflict

        if has_uv and has_vu:
            if TsetlinMBElbowContinuousRegressor._would_create_cycle(dag, u, v):
                return False  # would close a directed cycle -> conflict
            dag.remove_edge(v, u)
            return True

        return False  # already oriented u -> v, nothing changed

    def _apply_meek_rules(self, K, colliders):
        """
        Turns skeleton K + verified colliders into a CPDAG: orient the
        v-structures, then propagate further orientations with Meek's
        rules (R1-R3) until a fixpoint. Edges still ambiguous afterwards
        stay undirected (both arcs present in the returned DiGraph).
        """
        dag = nx.DiGraph()
        for u, v in K.edges():
            dag.add_edge(u, v)
            dag.add_edge(v, u)

        for X, Y, T in colliders:
            self._orient_edge(dag, X, Y)
            self._orient_edge(dag, T, Y)

        def adjacent(a, b):
            return dag.has_edge(a, b) or dag.has_edge(b, a)

        def oriented_edges():
            return [(u, v) for u, v in dag.edges() if not dag.has_edge(v, u)]

        def undirected_pairs():
            seen = set()
            pairs = []
            for u, v in dag.edges():
                if dag.has_edge(v, u):
                    key = frozenset((u, v))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((u, v))
            return pairs

        changed = True
        while changed:
            changed = False

            # R1: a -> b, b - c undirected, a & c not adjacent => b -> c
            # (orienting c -> b instead would create an unverified collider a -> b <- c)
            for a, b in oriented_edges():
                for c in list(dag.successors(b)):
                    if c == a or not dag.has_edge(c, b):
                        continue  # b - c must currently be undirected
                    if adjacent(a, c):
                        continue
                    if self._orient_edge(dag, b, c):
                        changed = True

            # R2: a -> b -> c, a - c undirected => a -> c (avoid a cycle)
            for a, b in oriented_edges():
                for c in list(dag.successors(b)):
                    if c == a or dag.has_edge(c, b):
                        continue  # need b -> c oriented, not undirected
                    if dag.has_edge(a, c) and dag.has_edge(c, a):
                        if self._orient_edge(dag, a, c):
                            changed = True

            # R3: a-b, a-c, a-d undirected; c->b, d->b oriented; c,d not adjacent => a -> b
            for a, b in undirected_pairs():
                neighbors_a = [n for n in dag.successors(a) if dag.has_edge(n, a) and n != b]
                candidates = [c for c in neighbors_a if dag.has_edge(c, b) and not dag.has_edge(b, c)]
                for c, d in itertools.combinations(candidates, 2):
                    if not adjacent(c, d):
                        if self._orient_edge(dag, a, b):
                            changed = True
                        break

        return dag

    def run(self, samples, raw_samples):
        start_time = time.time()
        one_encoded_samples = pd.get_dummies(samples).astype(int)
        one_hot_columns = one_encoded_samples.columns.tolist()
        target_columns = raw_samples.columns.tolist()
        output_arrays_1 = []

        # Train a Regression TM per target column, directly on the
        # continuous value - no quantile binning of the target.
        for target_column in target_columns:
            related_columns = [col for col in one_hot_columns if col.startswith(target_column + "_")]

            X = one_encoded_samples.drop(columns=related_columns)
            y = raw_samples[target_column].astype(np.float64)

            if y.nunique() < 2:
                # Constant target column - nothing for a regressor to learn.
                print(f"Target Column: {target_column}")
                print("Skipped: constant column (no variation)\n")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train_np = X_train.to_numpy().astype(np.uint32)
            X_test_np = X_test.to_numpy().astype(np.uint32)
            y_train_np = y_train.to_numpy()
            y_test_np = y_test.to_numpy()

            tm = TMRegressor(
                number_of_clauses=self.number_clauses,
                T=self.T,
                s=self.s, # specificity
                platform="CPU",
                weighted_clauses=True,
            )

            for epoch in range(self.num_epochs):
                tm.fit(X_train_np, y_train_np)

            learned_literals = tm.clause_bank.get_literals()
            clause_weights = tm.weight_bank.get_weights()

            for i in range(self.number_clauses):
                clause_literals = learned_literals[i][:len(X.columns) * 2]
                relevant_features = [
                    feature for feature, lit in zip(X.columns.tolist() * 2, clause_literals) if lit != 0
                ]
                output_variable = target_column
                output_arrays_1.append([*relevant_features, output_variable, clause_weights[i]])

            pred = tm.predict(X_test_np)
            mse = np.mean((pred - y_test_np) ** 2)
            baseline_mse = np.mean((y_test_np - y_train_np.mean()) ** 2)
            # Correlation is undefined if either side is constant - guard both,
            # not just pred, or corrcoef divides by a zero stddev on the
            # y_test side and raises RuntimeWarning: invalid value in divide
            # (common for CT Slice's sparse/near-constant histogram-bin columns).
            corr = (
                np.corrcoef(pred, y_test_np)[0, 1]
                if np.std(pred) > 0 and np.std(y_test_np) > 0
                else float("nan")
            )

            print(f"Target Column: {target_column}")
            print(f"MSE: {mse:.4f} (baseline: {baseline_mse:.4f})  Corr: {corr:.3f}\n")

        # Clean and process clauses
        cleaned_output = [self.clean_and_deduplicate(sublist[:-2]) + [sublist[-2], abs(sublist[-1])] for sublist in output_arrays_1]

        target_candidates = self.calculate_input_output_frequency_weightage(cleaned_output, self.top_n)

        # Build undirected graph K (skeleton)

        K = nx.Graph()
        separating_sets = {}

        for target, candidates_with_scores in target_candidates.items():
            PC = self.compute_target_PC(target, candidates_with_scores, raw_samples, separating_sets)
            for neighbor in PC:
                K.add_edge(target, neighbor)

        # Phase II (MMMB idea): find spouses via unshielded colliders X -> Y <- T.
        # See wtm_bn.py for the full rationale behind this hybrid
        # sepset-reuse strategy - unchanged here except the CI test runs on
        # raw_samples.
        colliders = []                 # list of (X, Y, T) meaning X -> Y <- T
        spouses = defaultdict(set)     # node -> set of spouse nodes (MB = PC U spouses)
        seen_triples = set()

        for T in K.nodes():
            pc_T = set(K.neighbors(T))

            for Y in pc_T:
                pc_Y = set(K.neighbors(Y))

                for X in pc_Y:
                    if X == T or X in pc_T:
                        continue

                    triple_key = (frozenset((X, T)), Y)
                    if triple_key in seen_triples:
                        continue
                    seen_triples.add(triple_key)

                    sepset_X_T = separating_sets.get(self._sepset_key(X, T))

                    if sepset_X_T is None:
                        candidates = [n for n in pc_T if n != Y and n != X]
                        for r in range(0, min(3, len(candidates)) + 1):
                            for S in itertools.combinations(candidates, r):
                                if self.ci_test_counter(X, T, list(S), raw_samples):
                                    sepset_X_T = list(S)
                                    break
                            if sepset_X_T is not None:
                                break

                    if sepset_X_T is None:
                        # Couldn't establish a baseline X,T independence at all ->
                        # not a valid collider candidate, skip.
                        continue

                    if Y in sepset_X_T:
                        continue

                    is_independent = self.ci_test_counter(
                        X, T, list(sepset_X_T) + [Y], raw_samples
                    )

                    if not is_independent:
                        # X, T become dependent once Y is added -> unshielded collider
                        colliders.append((X, Y, T))
                        spouses[T].add(X)
                        spouses[X].add(T)

        cpdag = self._apply_meek_rules(K, colliders)

        end_time = time.time()
        runtime = end_time - start_time

        return {
            "model": K,
            "cpdag": cpdag,
            "runtime": runtime,
            "ci_tests": self.ci_test_count,
            "colliders": colliders,
            "spouses": dict(spouses),
            "separating_sets": separating_sets,
        }
