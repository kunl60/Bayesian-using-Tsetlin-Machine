import networkx as nx


def general_graph_to_digraph(general_graph, node_names):
    """
    Convert a causal-learn GeneralGraph into a networkx DiGraph: a fully
    resolved edge (tail at one end, arrow at the other) appears as a single
    arc; any other mark combination - CPDAG's undirected i --- j, or PAG
    marks involving a circle endpoint (i o-o j, i o-> j, etc.) - appears as
    both u->v and v->u, since the direction isn't determined.

    causal-learn encodes endpoints in `general_graph.graph` (Tetrad style):
    -1 = tail, 1 = arrow, 2 = circle. graph[i, j] == -1 and graph[j, i] == 1
    means i --> j. `node_names` must be in the same order the graph was
    built with.
    """
    matrix = general_graph.graph
    dag = nx.DiGraph()
    dag.add_nodes_from(node_names)

    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                dag.add_edge(node_names[i], node_names[j])
            elif matrix[i, j] == 1 and matrix[j, i] == -1:
                dag.add_edge(node_names[j], node_names[i])
            elif matrix[i, j] != 0 and matrix[j, i] != 0:
                dag.add_edge(node_names[i], node_names[j])
                dag.add_edge(node_names[j], node_names[i])

    return dag
