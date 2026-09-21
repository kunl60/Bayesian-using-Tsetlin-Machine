import time

import networkx as nx
from xges import XGES


class XGESAlgorithm:
    """
    Wrapper for XGES (Extremely Greedy Equivalence Search, Nazaret & Blei,
    UAI 2024) - the pure-Python implementation from
    https://github.com/ANazaret/XGES (`pip install xges`), transparently
    numba-accelerated when numba is installed. Improves on GES with a more
    efficient search strategy and, with extended_search=True (XGES rather
    than XGES-0), a more accurate one. Continuous data only: like
    BOSSAlgorithm's "local_score_BIC_from_cov" and DGESAlgorithm, its BIC
    score is covariance-based (Gaussian), with no discrete/BDeu path. Same
    "model"/"cpdag" contract as the other wrappers: an undirected networkx
    Graph for "model" and a networkx-DiGraph-like CPDAG (unresolved edges
    appear as both u->v and v->u) for "cpdag" - xges's own PDAG.to_networkx()
    already encodes undirected edges that way, so only relabeling its
    integer node ids back to the original column names is needed.

    alpha is XGES's own BIC penalty term (not the same knob as
    significance_level in the constraint-based wrappers): higher alpha
    penalizes complexity more, yielding sparser graphs. 2.0 is XGES's own
    default.
    """

    def __init__(self, alpha=2.0, extended_search=True, use_fast_numba=True):
        self.alpha = alpha
        self.extended_search = extended_search
        self.use_fast_numba = use_fast_numba

    def run(self, data):
        columns = [str(c) for c in data.columns]
        array = data.to_numpy(dtype=float)

        start = time.time()
        pdag = XGES(alpha=self.alpha).fit(
            array,
            extended_search=self.extended_search,
            use_fast_numba=self.use_fast_numba,
            verbose=0,
        )
        runtime = time.time() - start

        cpdag = nx.relabel_nodes(pdag.to_networkx(), dict(enumerate(columns)))

        return {
            "model": cpdag.to_undirected(),
            "cpdag": cpdag,  # CPDAG, kept for direction metrics
            "runtime": runtime,
            "ci_tests": 0,  # Score-based methods don't use CI tests
        }
