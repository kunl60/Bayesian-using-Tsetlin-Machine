import time

from algorithms.cpc_lib.CPC import CPC
from algorithms.causallearn_utils import general_graph_to_digraph


class CPCAlgorithm:
    """
    Wrapper for CPC (a causal-learn based PC/FCI variant with kPC-style PAG
    orientation rules, vendored from https://github.com/... uai_cpc_code).
    Works on both categorical data (tester="chisq"/"gsq") and continuous
    data (tester="fisherz"/"kci"), depending on the CI test selected - same
    "model"/"cpdag" contract as BOSSAlgorithm. Runs in standard
    bounded-subset PC mode (traverse_subset_and_skip=True) rather than the
    externally-curated separating-set mode the original research code
    supports, so it doesn't require any domain-knowledge input.
    """

    def __init__(self, tester="chisq", alpha=0.01, max_cond_vars=3):
        self.tester = tester
        self.alpha = alpha
        self.max_cond_vars = max_cond_vars

    def run(self, data):
        columns = [str(c) for c in data.columns]
        array = data.to_numpy(dtype=float)

        start = time.time()
        general_graph, _ = CPC(
            array,
            tester=self.tester,
            I=[],
            n=array.shape[1],
            data_names=columns,
            alpha=self.alpha,
            traverse_subset_and_skip=True,
            k=self.max_cond_vars,
        )
        runtime = time.time() - start

        cpdag = general_graph_to_digraph(general_graph, columns)

        return {
            "model": cpdag.to_undirected(),
            "cpdag": cpdag,  # PAG (FCI/kPC-oriented), kept for direction metrics
            "runtime": runtime,
            "ci_tests": general_graph.ci_test_count,
        }
