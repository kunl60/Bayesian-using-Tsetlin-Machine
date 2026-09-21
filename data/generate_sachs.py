import networkx as nx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling


def load_sachs_dataset(sample_size=5000, seed=None):
    """
    Generate Sachs dataset via Bayesian sampling and preprocess it.

    `seed` controls the sampling RNG so repeated calls can draw different
    (but reproducible) samples from the same ground-truth model - used to
    run an algorithm over several seeds and report mean +/- std.
    """
    model = get_example_model("pigs")
    sampler = BayesianModelSampling(model)
    samples = sampler.rejection_sample(size=sample_size, seed=seed, show_progress=False)

    # Remove underscores from node names
    node_mapping = {node: node.replace('_', '') for node in model.nodes()}

    # Relabel nodes in the Bayesian model
    model = nx.relabel_nodes(model, node_mapping)

    # Rename columns in the samples DataFrame
    samples.rename(columns=node_mapping, inplace=True)

    # Map node names to numeric strings ("0", "1", "2", ...)
    nodes = list(model.nodes())
    numeric_mapping = {node: str(i) for i, node in enumerate(nodes)}
    model = nx.relabel_nodes(model, numeric_mapping)
    samples.rename(columns=numeric_mapping, inplace=True)


    # Label encode categorical variables
    le = LabelEncoder()
    encoded_samples = samples.copy()
    for col in encoded_samples.columns:
        if not pd.api.types.is_numeric_dtype(encoded_samples[col]):
            encoded_samples[col] = le.fit_transform(encoded_samples[col].astype(str))

    return encoded_samples, model, samples


