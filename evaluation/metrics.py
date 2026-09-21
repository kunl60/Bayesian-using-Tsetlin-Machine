def get_nodes_difference(true_graph, pred_graph):
    """
    Calculates the absolute difference in the number of nodes.
    """
    return abs(len(true_graph.nodes) - len(pred_graph.nodes))

def structural_hamming_distance(true_graph, pred_graph):
    true_edges = {tuple(sorted(e)) for e in true_graph.edges()}
    pred_edges = {tuple(sorted(e)) for e in pred_graph.edges()}

    false_positives = len(pred_edges - true_edges)
    false_negatives = len(true_edges - pred_edges)

    shd = false_positives + false_negatives
    return shd, false_positives, false_negatives


def precision_recall_f1(true_graph, pred_graph):
    true_edges = {tuple(sorted(e)) for e in true_graph.edges()}
    pred_edges = {tuple(sorted(e)) for e in pred_graph.edges()}

    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, f1


def ci_tests_per_correct_edge(ci_tests, true_graph, pred_graph):
    """
    Normalizes CI test count by the number of correctly recovered edges (true positives).
    Lower is better — measures how many CI tests it costs to correctly find one edge.
    Returns float('inf') if no edges were correctly recovered.
    """
    true_edges = {tuple(sorted(e)) for e in true_graph.edges()}
    pred_edges = {tuple(sorted(e)) for e in pred_graph.edges()}
    tp = len(true_edges & pred_edges)
    return ci_tests / tp if tp > 0 else float('inf')


def directed_precision_recall_f1(true_graph, pred_cpdag):
    """
    Directed precision/recall/F1 over a predicted CPDAG (a DiGraph where an
    unresolved edge appears as both u->v and v->u). Only arcs the CPDAG has
    actually resolved to a single direction count as predictions - an edge
    left undirected contributes a false negative for whichever direction is
    true, the same as a missing edge would.
    """
    true_edges = set(true_graph.edges())
    pred_edges = {
        (u, v) for u, v in pred_cpdag.edges() if not pred_cpdag.has_edge(v, u)
    }

    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, f1


def orientation_accuracy(true_graph, pred_cpdag):
    """
    Among skeleton edges correctly recovered (present in both the true
    graph and the predicted CPDAG, direction ignored), how many did the
    collider + Meek's-rules step orient correctly?

    Returns (correct, incorrect, undirected, accuracy), where accuracy is
    computed only over resolved edges (correct + incorrect) - it isolates
    the direction step's accuracy from skeleton recall, which the existing
    precision/recall/F1 already covers.
    """
    true_skeleton = {frozenset(e) for e in true_graph.edges()}
    pred_skeleton = {frozenset((u, v)) for u, v in pred_cpdag.edges()}
    recovered = true_skeleton & pred_skeleton

    correct = 0
    incorrect = 0
    undirected = 0

    for edge in recovered:
        u, v = tuple(edge)
        if not true_graph.has_edge(u, v):
            u, v = v, u  # orient (u, v) so true_graph.has_edge(u, v) is True

        pred_uv = pred_cpdag.has_edge(u, v)
        pred_vu = pred_cpdag.has_edge(v, u)

        if pred_uv and not pred_vu:
            correct += 1
        elif pred_vu and not pred_uv:
            incorrect += 1
        else:
            undirected += 1

    resolved = correct + incorrect
    accuracy = correct / resolved if resolved else 0

    return correct, incorrect, undirected, accuracy
