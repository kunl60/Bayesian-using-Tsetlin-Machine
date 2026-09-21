import time
import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
from algorithms.ci_tests import chi_square


class MBElbowBase:

    def __init__(self, top_n, alpha=0.01, maxK=3):
        self.top_n = top_n
        self.alpha = alpha
        self.maxK = maxK
        self.ci_test_count = 0

    def _score_candidates(self, label_encoded_samples, samples):
        """
        Return dict[target] = {candidate: score}, one entry per column
        of label_encoded_samples with >= 2 observed classes, scoring
        every other column's association with it. Higher score = more
        associated. Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def find_elbow_index(scores):
        if len(scores) <= 2:
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
        # We want the largest relative drop -> most negative value
        elbow_index = np.argmin(relative_diffs) + 1

        return elbow_index

    def _select_with_elbow(self, raw_scores):
        """
        raw_scores: dict[target] = {candidate: score}.
        Returns dict[target] = [(candidate, score), ...], sorted
        descending and cut at the elbow (never fewer than top_n).
        """
        result = {}
        selected_counts = {}

        for target, candidate_scores in raw_scores.items():
            scored_inputs = sorted(
                candidate_scores.items(), key=lambda kv: kv[1], reverse=True
            )
            scores = [score for _, score in scored_inputs]

            elbow_idx = self.find_elbow_index(scores)

            selected_count = max(elbow_idx, self.top_n)
            selected_count = min(selected_count, len(scored_inputs))
            selected_counts[target] = selected_count

            result[target] = scored_inputs[:selected_count]

        print(f"selected_count per target: {selected_counts}")
        print(f"selected_count list: {list(selected_counts.values())}")

        return result

    def chi_square_counter(self, X, Y, Z, data):
        self.ci_test_count += 1
        return chi_square(
            X=X, Y=Y, Z=Z, data=data,
            boolean=True,
            significance_level=self.alpha
        )

    @staticmethod
    def _sepset_key(a, b):
        return tuple(sorted((a, b)))

    def compute_target_PC(self, target, candidates_with_scores, label_encoded_samples, separating_sets):
    
        score = dict(candidates_with_scores)
        PC = []
        for X in score:
            if self.chi_square_counter(X, target, [], label_encoded_samples):
                separating_sets[self._sepset_key(X, target)] = []
            else:
                PC.append(X)

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
                    if self.chi_square_counter(X, target, Z, label_encoded_samples):
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
            if MBElbowBase._would_create_cycle(dag, u, v):
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

        raw_scores = self._score_candidates(label_encoded_samples, samples)
        target_candidates = self._select_with_elbow(raw_scores)

        # Build undirected graph K (skeleton)
        K = nx.Graph()
        separating_sets = {}

        for target, candidates_with_scores in target_candidates.items():
            PC = self.compute_target_PC(target, candidates_with_scores, label_encoded_samples, separating_sets)
            for neighbor in PC:
                K.add_edge(target, neighbor)

        # Phase II (MMMB idea): find spouses via unshielded colliders X -> Y <- T.
        # PC(T) and PC(Y) are just K.neighbors(...) - K is already the merged
        # PC structure from Phase I, so there's no need to re-run MMPC per node.
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
                        for r in range(0, min(3, len(candidates)) + 1):
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
        }
