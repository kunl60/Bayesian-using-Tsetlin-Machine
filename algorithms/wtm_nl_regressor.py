import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tmu.models.regression.vanilla_regressor import TMRegressor
import networkx as nx
from collections import defaultdict
import time
import itertools
from algorithms.ci_tests import fisher_z
import csv

class TsetlinMBContinuousRegressor:

    def __init__(self, num_epochs, number_clauses, top_n, alpha=0.01, maxK=3, T=10, s=1):
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
    def calculate_input_output_frequency_weightage(cleaned_output, top_n):
        output_data = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "total_weightage": 0}))

        for entry in cleaned_output:
            *inputs, output, weight = entry

            for input_element in inputs:
                output_data[output][input_element]["frequency"] += 1
                output_data[output][input_element]["total_weightage"] += weight

        result = []

        for output, input_data in output_data.items():
            sorted_inputs = sorted(input_data.items(), key=lambda x: x[1]["frequency"] * x[1]["total_weightage"], reverse=True)

            top_inputs = sorted_inputs[:top_n]

            result.append([input for input, _ in top_inputs] + [output])

        return result

    @staticmethod
    def _sepset_key(a, b):
        return tuple(sorted((a, b)))

    def ci_test_counter(self, X, Y, Z, data):
        self.ci_test_count += 1
        return fisher_z(
            X=X, Y=Y, Z=Z, data=data,
            boolean=True,
            significance_level=self.alpha
        )

    def compute_target_PC(self, target, top_selected, raw_samples, separating_sets):
        """
        Two-sided, shrinking PC(target) search - ported from wtm_nl.py, see
        that file's compute_target_PC docstring for the full rationale.
        Same design, just running Fisher's Z on raw_samples instead of
        chi-square on label-encoded categorical data.
        """
        PC = list(top_selected.get(target, []))
        blacklist = set()

        for B in list(reversed(PC)):
            if B not in PC:
                continue  # already removed earlier in this same pass

            own_remaining = [v for v in PC if v != B]
            partner_remaining = [v for v in top_selected.get(B, []) if v != target]

            seen = set()
            Z = []
            for v in own_remaining + partner_remaining:
                if v == B or v in blacklist or v in seen:
                    continue
                seen.add(v)
                Z.append(v)

            if self.ci_test_counter(target, B, [], raw_samples):
                separating_sets[self._sepset_key(target, B)] = []
                PC.remove(B)
                blacklist.add(B)
                continue

            removed = False
            for r in range(1, min(self.maxK, len(Z)) + 1):
                for combo in itertools.combinations(Z, r):
                    if self.ci_test_counter(target, B, list(combo), raw_samples):
                        separating_sets[self._sepset_key(target, B)] = list(combo)
                        removed = True
                        break
                if removed:
                    break

            if removed:
                PC.remove(B)
                blacklist.add(B)

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
            if TsetlinMBContinuousRegressor._would_create_cycle(dag, u, v):
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
            corr = np.corrcoef(pred, y_test_np)[0, 1] if np.std(pred) > 0 else float("nan")

            print(f"Target Column: {target_column}")
            print(f"MSE: {mse:.4f} (baseline: {baseline_mse:.4f})  Corr: {corr:.3f}\n")

        # Clean and process clauses
        cleaned_output = [self.clean_and_deduplicate(sublist[:-2]) + [sublist[-2], abs(sublist[-1])] for sublist in output_arrays_1]

        unique_array = self.calculate_input_output_frequency_weightage(cleaned_output, self.top_n)

        top_selected = {arr[-1]: arr[:-1] for arr in unique_array}

        with open('insurance_5_top.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(unique_array)

        # --- Old version: built ONE merged Z per unordered pair from the
        # --- *static* top-lists, tested each pair once, first-match-stops.
        # --- Replaced below with a per-target, shrinking two-sided PC
        # --- search (compute_target_PC) - see wtm_nl.py for the full
        # --- rationale (same fix, ported here for the regressor).
        #
        # result = {}
        #
        # for arr in unique_array:
        #     output_element = arr[-1]
        #     input_elements = arr[:-1]
        #
        #     for input_element in input_elements:
        #
        #         pair = (output_element, input_element)
        #         reverse_pair = (input_element, output_element)
        #
        #         if reverse_pair in result:
        #             continue
        #
        #         remaining_elements = [el for el in arr if el != input_element and el != output_element]
        #
        #         for next_arr in unique_array:
        #             if next_arr[-1] == input_element:
        #
        #                 next_remaining_elements = [el for el in next_arr if el != input_element and el != next_arr[-1]]
        #
        #                 combined_remaining_elements = set(remaining_elements + next_remaining_elements)
        #                 combined_remaining_elements.discard(output_element)
        #                 combined_remaining_elements.discard(input_element)
        #
        #                 result[pair] = list(combined_remaining_elements)
        #
        # # Build undirected graph K (skeleton)
        #
        # K = nx.Graph()
        # separating_sets = {}
        #
        # for pair, Z in result.items():
        #     X, Y = pair
        #
        #     # Initial independence test - run on the raw continuous data
        #     ci_test = self.ci_test_counter(X, Y, [], raw_samples)
        #
        #     if not ci_test:
        #
        #         independent = False
        #         found_sepset = None
        #         for r in range(1, min(3, len(Z)) + 1):
        #             for z_combination in itertools.combinations(Z, r):
        #                 test_with_Z = self.ci_test_counter(X, Y, list(z_combination), raw_samples)
        #
        #                 if test_with_Z:
        #                     independent = True
        #                     found_sepset = list(z_combination)
        #                     break
        #             if independent:
        #                 break
        #
        #         if not independent:
        #             K.add_edge(X, Y)
        #         else:
        #             separating_sets[self._sepset_key(X, Y)] = found_sepset
        #
        #     else:
        #         # X, Y already independent given the empty set
        #         separating_sets[self._sepset_key(X, Y)] = []

        # Build undirected graph K (skeleton)

        K = nx.Graph()
        separating_sets = {}

        for target in top_selected:
            PC = self.compute_target_PC(target, top_selected, raw_samples, separating_sets)
            for neighbor in PC:
                K.add_edge(target, neighbor)

        # Phase II (MMMB idea): find spouses via unshielded colliders X -> Y <- T.
        # See wtm_nl.py for the full rationale behind this hybrid sepset-reuse
        # strategy - unchanged here except the CI test runs on raw_samples.
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
                        for r in range(0, min(self.maxK, len(candidates)) + 1):
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
