import time
import warnings

from causallearn.search.PermutationBased.BOSS import boss

from algorithms.causallearn_utils import general_graph_to_digraph


class BOSSAlgorithm:
    """
    Wrapper for the Best Order Score Search (BOSS) permutation-based
    causal discovery algorithm (causal-learn). Works on both discrete
    data (score_func="local_score_BDeu") and continuous data
    (score_func="local_score_BIC"/"local_score_BIC_from_cov", the default),
    depending on the score function selected.
    """

    def __init__(self, score_func="local_score_BIC_from_cov", parameters=None):
        self.score_func = score_func
        self.parameters = parameters

    def run(self, data):
        columns = [str(c) for c in data.columns]

        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module="causallearn.score.LocalScoreFunction",
            )
            general_graph = boss(
                data.values.astype(float),
                score_func=self.score_func,
                parameters=self.parameters,
                verbose=False,
                node_names=columns,
            )
        runtime = time.time() - start

        cpdag = general_graph_to_digraph(general_graph, columns)

        return {
            "model": cpdag.to_undirected(),
            "cpdag": cpdag,  # CPDAG, kept for direction metrics
            "runtime": runtime,
            "ci_tests": 0,  # Score-based methods don't use CI tests
        }
