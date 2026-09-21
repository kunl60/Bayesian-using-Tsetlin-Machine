import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.utils import get_example_model


def _thermometer_encode(raw_samples, n_thresholds=10):
    """
    Threshold ("thermometer") encoding of raw continuous samples: each
    column gets up to `n_thresholds` quantile cut points, and each cut point
    becomes its own boolean literal - "<col>_t<i>" = 1 if the value exceeds
    the i-th threshold, else 0.

    This replaces one-hot quantile-bin encoding (pd.qcut + pd.get_dummies)
    as the TM front end's predictor representation. One-hot bins have no
    notion of order - "bin 1" and "bin 5" look exactly as different to a
    Tsetlin Machine as "bin 2" and "bin 3" do - so a weak/moderate linear
    trend (common in ecoli70's linear-Gaussian edges) gets hidden inside
    whichever single bucket a value lands in. Thermometer encoding keeps a
    value's literal pattern monotonic (crossing threshold i implies crossing
    every threshold below it), so TM's clauses can still pick up on
    magnitude/trend instead of only "which disconnected bucket" - see
    TsetlinMBElbowGESContinuous's docstring for the fuller rationale.

    Column names keep the "<col>_..." convention the TM front ends already
    rely on (they strip everything after the first "_" to recover the
    original variable name, and drop a target's own "<col>_*" columns before
    training), and the output is already 0/1 int - pd.get_dummies leaves
    int/bool columns untouched, so this passes straight through as a
    drop-in replacement for the old one-hot categorical_samples.
    """
    encoded = {}
    for col in raw_samples.columns:
        values = raw_samples[col].astype(float)
        quantile_points = np.linspace(0, 1, n_thresholds + 2)[1:-1]
        thresholds = np.unique(values.quantile(quantile_points).to_numpy())
        for i, t in enumerate(thresholds):
            encoded[f"{col}_t{i}"] = (values > t).astype(int)
    return pd.DataFrame(encoded, index=raw_samples.index)


def load_ecoli_dataset(sample_size=5000, n_bins=10, n_thresholds=10, seed=42):
    """
    Generate the ECOLI70 continuous (linear-Gaussian) dataset via sampling
    and preprocess it into both a continuous and a binned/categorical form.

    Returns
    -------
    raw_samples : pd.DataFrame
        Continuous samples, unmodified - feed this directly to a
        continuous-native CI test (e.g. fisher_z) or PC-continuous.
    ground_truth : nx.DiGraph
        The true DAG, with nodes relabeled to numeric strings ("0", "1", ...)
        to match the convention used by load_sachs_dataset.
    categorical_samples : pd.DataFrame
        Thermometer/threshold-encoded version of raw_samples (see
        `_thermometer_encode`) - `n_thresholds` boolean columns per
        variable, already suitable as-is for pd.get_dummies. This is the TM
        front end's predictor input; it no longer holds actual category
        labels (name kept for call-site compatibility - every consumer only
        ever forwards it into pd.get_dummies).
    binned_samples : pd.DataFrame
        Label-encoded (integer) classification target per node, `n_bins`
        quantile classes - unaffected by the predictor-side change above,
        since a classifier target still needs a handful of discrete classes
        rather than threshold literals.
    """
    model = get_example_model("ecoli70")
    raw_samples = model.simulate(n_samples=sample_size, seed=seed)

    # Map node names to numeric strings ("0", "1", ...), same as generate_sachs.
    nodes = list(model.nodes())
    numeric_mapping = {node: str(i) for i, node in enumerate(nodes)}
    ground_truth = nx.relabel_nodes(model, numeric_mapping)
    raw_samples = raw_samples.rename(columns=numeric_mapping)

    # --- Previous predictor encoding: one-hot quantile bins -----------------
    # Fed straight into pd.get_dummies as the TM's Boolean input space - same
    # role thermometer encoding fills below. Left here for reference: it was
    # starving the GES/PC candidate whitelist of true parents on continuous
    # data, because collapsing a value into a single disconnected bucket
    # hides weak/moderate linear trends from the TM entirely.
    #
    # bin_labels = [f"bin{i}" for i in range(n_bins)]
    # categorical_samples = raw_samples.apply(
    #     lambda col: pd.qcut(col, q=n_bins, labels=bin_labels, duplicates="drop")
    # ).astype(str)
    # -------------------------------------------------------------------------

    categorical_samples = _thermometer_encode(raw_samples, n_thresholds=n_thresholds)

    # Classification targets still need a small number of discrete classes,
    # not threshold literals - keep deriving these from plain qcut bins.
    bin_labels = [f"bin{i}" for i in range(n_bins)]
    quantile_bins = raw_samples.apply(
        lambda col: pd.qcut(col, q=n_bins, labels=bin_labels, duplicates="drop")
    ).astype(str)

    le = LabelEncoder()
    binned_samples = quantile_bins.apply(lambda col: le.fit_transform(col))

    return raw_samples, ground_truth, categorical_samples, binned_samples
