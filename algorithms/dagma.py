import time
import networkx as nx
from dagma.linear import DagmaLinear


class DagmaAlgorithm:
    """
    Wrapper for DAGMA (Bello, Aragam & Ravikumar, NeurIPS'22) - a log-det
    acyclicity reformulation of NOTEARS for continuous linear-Gaussian data.
    """

    def __init__(self, loss_type="l2", lambda1=0.3, w_threshold=0.3, T=5):
        self.loss_type = loss_type
        self.lambda1 = lambda1
        self.w_threshold = w_threshold
        self.T = T

    def run(self, data):
        columns = [str(c) for c in data.columns]

        # w_threshold is a fixed cutoff on the fitted coefficients, so
        # unstandardized columns (which can differ in scale by an order of
        # magnitude or more) would have edges pruned or kept based on scale
        # rather than actual relationship strength.
        standardized = (data - data.mean()) / data.std()

        model = DagmaLinear(loss_type=self.loss_type)

        start = time.time()
        W_est = model.fit(
            standardized.values.astype(float),
            lambda1=self.lambda1,
            w_threshold=self.w_threshold,
            T=self.T,
        )
        runtime = time.time() - start

        # W_est is a weighted adjacency matrix, already thresholded:
        # entry [i, j] != 0 means an edge i -> j.
        dag = nx.DiGraph()
        dag.add_nodes_from(columns)
        for i, u in enumerate(columns):
            for j, v in enumerate(columns):
                if W_est[i, j] != 0:
                    dag.add_edge(u, v)

        return {
            "model": dag.to_undirected(),
            "cpdag": dag,  # fully-oriented DAG, kept for direction metrics
            "runtime": runtime,
            "ci_tests": 0,  # Gradient-based methods don't use CI tests
        }
