import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tmu.models.classification.vanilla_classifier import TMClassifier
import networkx as nx
from collections import defaultdict
import time
import itertools
from algorithms.ci_tests import chi_square
import csv

class TsetlinMB:
    """
    Tsetlin Machine based Markov Blanket discovery
    """

    def __init__(self, num_epochs, number_clauses, top_n, alpha=0.01, maxK=3, T=10, s=5):
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
        
        #print("Frequency and Weightage for each Input-Output Pair:")
        for output, input_data in output_data.items():
            for input_element, values in input_data.items():
                frequency = values["frequency"]
                total_weightage = values["total_weightage"]
                #print(f"Output: {output}, Input: {input_element}, Frequency: {frequency}, Total Weightage: {total_weightage} = {frequency + total_weightage}")

        result = []

        for output, input_data in output_data.items():
            sorted_inputs = sorted(input_data.items(), key=lambda x: x[1]["frequency"], reverse=True)
            
            top_inputs = sorted_inputs[:top_n]

            result.append([input for input, _ in top_inputs] + [output])

        return result
    
    
    @staticmethod
    def _sepset_key(a, b):
        return tuple(sorted((a, b)))

    def compute_target_PC(self, target, top_selected, label_encoded_samples, separating_sets):
        PC = list(top_selected.get(target, []))
        blacklist = set()

        for B in list(reversed(PC)):
            if B not in PC:
                continue  # already removed earlier in this same pass

            own_remaining = [v for v in PC if v != B]
            partner_remaining = [v for v in top_selected.get(B, []) if v != target]

            # Order-preserving merge+dedup (own candidates first, both
            # sides already strongest-first) - NOT a set(), whose
            # iteration order is arbitrary and was the original
            # algorithm's random-order weakness.
            seen = set()
            Z = []
            for v in own_remaining + partner_remaining:
                if v == B or v in blacklist or v in seen:
                    continue
                seen.add(v)
                Z.append(v)

            if self.chi_square_counter(target, B, [], label_encoded_samples):
                separating_sets[self._sepset_key(target, B)] = []
                PC.remove(B)
                blacklist.add(B)
                continue

            removed = False
            for r in range(1, min(self.maxK, len(Z)) + 1):
                for combo in itertools.combinations(Z, r):
                    if self.chi_square_counter(target, B, list(combo), label_encoded_samples):
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
    def _balance_to_fixed_size(X, y, target_size, random_state=42):
        """
        Rebalance classes in (X, y) by per-class resampling (with
        replacement where a class is short of its target, without
        replacement otherwise) so the total row count stays exactly
        target_size - minority classes get upsampled, majority classes
        get trimmed, keeping training time roughly unchanged.
        """
        classes = y.unique()
        n_classes = len(classes)
        base = target_size // n_classes
        remainder = target_size % n_classes

        parts = []
        for i, cls in enumerate(classes):
            cls_target = base + (1 if i < remainder else 0)
            mask = y == cls
            part = pd.concat([X[mask], y[mask]], axis=1)
            replace = len(part) < cls_target
            parts.append(resample(
                part,
                replace=replace,
                n_samples=cls_target,
                random_state=random_state,
            ))

        balanced = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
        return balanced[X.columns], balanced[y.name]

    def chi_square_counter(self, X, Y, Z, data):
        self.ci_test_count += 1
        return chi_square(
            X=X, Y=Y, Z=Z, data=data,
            boolean=True,
            significance_level=self.alpha
        )

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
            if TsetlinMB._would_create_cycle(dag, u, v):
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

    def run(self, label_encoded_samples, samples):
        start_time = time.time()
        one_encoded_samples = pd.get_dummies(samples).astype(int)
        one_hot_columns = one_encoded_samples.columns.tolist()
        label_columns = label_encoded_samples.columns.tolist()
        output_arrays_1 = []
        output_arrays = []

        # Train Tsetlin classifier per target column
        for target_column in label_columns:
            related_columns = [col for col in one_hot_columns if col.startswith(target_column + "_")]

            X = one_encoded_samples.drop(columns=related_columns)
            y = label_encoded_samples[target_column]
            n_classes = y.nunique()

            if n_classes < 2:
                # Constant target column - nothing for a classifier to learn.
                print(f"Target Column: {target_column}")
                print("Skipped: constant column (only one observed class)\n")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
            )

            X_train, y_train = self._balance_to_fixed_size(
                X_train, y_train, target_size=len(X_train)
            )

            X_train = X_train.astype(np.uint32)
            y_train = y_train.astype(np.uint32)
            X_test = X_test.astype(np.uint32)
            y_test = y_test.astype(np.uint32)

            tm = TMClassifier(
                number_of_clauses=self.number_clauses,
                T=self.T, # Threshold
                s=self.s, # specificaity
                platform="CPU",
                weighted_clauses=True,
                #use_minimal_feedback=False
            )

            learned_clauses_all_epochs = []  
        # number of Epochs
            for epoch in range(self.num_epochs):
                tm.fit(X_train.to_numpy(), y_train.to_numpy())
                
                
                if (epoch + 1) == self.num_epochs:
                
                    # as per the number of clauses used
                    for i in range(self.number_clauses):
                    # as per max unique values
                        for j in range(n_classes):
                                learned_clauses = tm.clause_banks[j].get_literals()[i][:len(X.columns) * 2] 
                                relevant_features = [feature for feature, clause in zip(X.columns.tolist() * 2, learned_clauses) if clause != 0]
                                output_variable = target_column
                                output_arrays.append([*relevant_features, output_variable])

                                weights_of_learned_clauses = tm.weight_banks[j].get_weights()[i] 
                                output_arrays_1.append([*relevant_features, output_variable, weights_of_learned_clauses])
                    
                result = 100 * (tm.predict(X_test.to_numpy()) == y_test.to_numpy()).mean()

                print(f"Target Column: {target_column}")
                print(f"Accuracy: {result:.2f}%\n")

        # Clean and process clauses
        cleaned_output = [self.clean_and_deduplicate(sublist[:-2]) + [sublist[-2], abs(sublist[-1])] for sublist in output_arrays_1]

        unique_array = self.calculate_input_output_frequency_weightage(cleaned_output, self.top_n)

        top_selected = {arr[-1]: arr[:-1] for arr in unique_array}

        with open('insurance_5_top.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(unique_array)

        # --- Old version: built ONE merged Z per unordered pair from the
        # --- *static* top-lists, tested each pair once, first-match-stops.
        # --- A variable already shown irrelevant to a target could still
        # --- get pulled back in (via the static merge) to spuriously
        # --- explain away a *different* candidate of that same target.
        # --- Replaced below with a per-target, shrinking two-sided PC
        # --- search (compute_target_PC) - see its docstring.
        #
        # result = {}
        #
        # for arr in unique_array:
        #     output_element = arr[-1]
        #     input_elements = arr[:-1]
        #
        #
        #     for input_element in input_elements:
        #
        #         pair = (output_element, input_element)
        #         reverse_pair = (input_element, output_element)
        #
        #
        #         if reverse_pair in result:
        #             continue
        #
        #         remaining_elements = [el for el in arr if el != input_element and el != output_element]
        #
        #
        #         for next_arr in unique_array:
        #             if next_arr[-1] == input_element:
        #
        #                 next_remaining_elements = [el for el in next_arr if el != input_element and el != next_arr[-1]]
        #
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
        #     # Initial independence test
        #     chi_square_test = self.chi_square_counter(X, Y, [], label_encoded_samples)
        #
        #     if not chi_square_test:
        #
        #         independent = False
        #         found_sepset = None
        #         for r in range(1, min(3, len(Z)) + 1):
        #             for z_combination in itertools.combinations(Z, r):
        #                 test_with_Z = self.chi_square_counter(X, Y, list(z_combination), label_encoded_samples)
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
            PC = self.compute_target_PC(target, top_selected, label_encoded_samples, separating_sets)
            for neighbor in PC:
                K.add_edge(target, neighbor)

        # Phase II (MMMB idea): find spouses via unshielded colliders X -> Y <- T.
        # PC(T) and PC(Y) are just K.neighbors(...) - K is already the merged
        # PC structure from Phase I, so there's no need to re-run MMPC per node
        # the way MMMB_G2 does.
        #
        # Original version reused Phase I's sepset(X,T) as a baseline before
        # testing Dep(X;T|Y). On this dataset separating_sets ends up too
        # sparse (Phase I only tests a small TM-proposed subset of pairs, so
        # "no sepset recorded" almost always means "never tested", not
        # "independent") - so the sepset requirement skipped nearly every
        # candidate triple and found 0 colliders. Kept here, commented out,
        # in case a denser candidate pool makes the rigorous version viable:
        #
        # for T in K.nodes():
        #     pc_T = set(K.neighbors(T))
        #     for Y in pc_T:
        #         pc_Y = set(K.neighbors(Y))
        #         for X in pc_Y:
        #             if X == T or X in pc_T:
        #                 continue
        #             sepset_X_T = separating_sets.get(self._sepset_key(X, T))
        #             if sepset_X_T is None:
        #                 continue
        #             if Y in sepset_X_T:
        #                 continue
        #             triple_key = (frozenset((X, T)), Y)
        #             if triple_key in seen_triples:
        #                 continue
        #             seen_triples.add(triple_key)
        #             is_independent = self.chi_square_counter(
        #                 X, T, list(sepset_X_T) + [Y], label_encoded_samples
        #             )
        #             if not is_independent:
        #                 colliders.append((X, Y, T))
        #                 spouses[T].add(X)
        #                 spouses[X].add(T)
        #
        # Simplified version (no baseline sepset at all, just Dep(X;T|Y)) was
        # tried too: cheap, but with no independence baseline it flagged
        # X,T as colliders even when they were dependent for reasons that
        # had nothing to do with Y - 19/30 resolved edges came out
        # mis-oriented. Kept here for reference, commented out:
        #
        # for T in K.nodes():
        #     pc_T = set(K.neighbors(T))
        #     for Y in pc_T:
        #         pc_Y = set(K.neighbors(Y))
        #         for X in pc_Y:
        #             if X == T or X in pc_T:
        #                 continue
        #             triple_key = (frozenset((X, T)), Y)
        #             if triple_key in seen_triples:
        #                 continue
        #             seen_triples.add(triple_key)
        #             is_independent = self.chi_square_counter(
        #                 X, T, [Y], label_encoded_samples
        #             )
        #             if not is_independent:
        #                 colliders.append((X, Y, T))
        #                 spouses[T].add(X)
        #                 spouses[X].add(T)
        #
        # Hybrid version: reuse Phase I's sepset(X,T) when we have it; when
        # we don't (pair was never tested in Phase I), search for one on
        # demand within T's other neighbors (bounded by the skeleton, same
        # idea PC/MMPC use to avoid searching the whole variable space) up
        # to size 3. Only if a genuine baseline independence is found do we
        # test whether adding Y breaks it.
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
                                if self.chi_square_counter(X, T, list(S), label_encoded_samples):
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

                    is_independent = self.chi_square_counter(
                        X, T, list(sepset_X_T) + [Y], label_encoded_samples
                    )

                    if not is_independent:
                        # X, T become dependent once Y is added -> unshielded collider
                        colliders.append((X, Y, T))
                        spouses[T].add(X)
                        spouses[X].add(T)

        cpdag = self._apply_meek_rules(K, colliders)

        runtime = time.time() - start_time

        return {
            "model": K,
            "cpdag": cpdag,
            "runtime": runtime,
            "ci_tests": self.ci_test_count,
            "colliders": colliders,
            "spouses": dict(spouses),
            "separating_sets": separating_sets,
            "top_selected": top_selected,
        }
