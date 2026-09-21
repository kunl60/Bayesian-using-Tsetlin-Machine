"""
Real-world fully-categorical datasets with heterogeneous per-column domain
sizes (unlike data/neticusdroid.csv, whose 86 columns are all uniform binary
permission flags) - for Markov-blanket/skeleton discovery on discrete data
with genuinely varying cardinality per variable, same role as neticusdroid.csv
in experiments_real_categorical/. No known ground-truth DAG for either, so
they're for MB/skeleton discovery and downstream classifier evaluation only,
not precision/recall against a true graph (mirrors generate_regression.py's
real-world continuous loaders in that respect).

OpenML-backed loaders download via sklearn's fetch_openml on first use; the
raw frame is cached to a local CSV in this directory (mirrors
generate_regression.py's caching) so repeated runs across the many
per-algorithm scripts don't re-hit the network. Each loader takes its own
default `sample_size` (None = use every row; an int row-subsamples, or uses
every row if the source has fewer than requested - "use whole samples"
rather than erroring).
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).parent


def _load_openml_frame(cache_file, **fetch_kwargs):
    cache_path = DATA_DIR / cache_file
    if cache_path.exists():
        return pd.read_csv(cache_path)

    bunch = fetch_openml(as_frame=True, parser="auto", **fetch_kwargs)
    df = bunch.frame.copy()
    df.columns = [str(c) for c in df.columns]
    df.to_csv(cache_path, index=False)
    return df


def _prepare_categorical(df, sample_size, seed):
    # Same column-name cleanup neticusdroid.csv's consumers apply inline
    # (strip everything but alphanumerics, lowercase) - keeps naming
    # consistent across every experiments_real_categorical/ script.
    df = df.dropna()
    df = df.rename(columns=lambda c: str(c))
    df.columns = (
        df.columns.str.replace(r"[^A-Za-z0-9]", "", regex=True).str.lower()
    )

    # Some sources arrive already integer-coded (coil2000, neticusdroid,
    # tuandromd) - keep those codes as-is. Others (mushroom) are letter
    # codes ('x', 's', 'n', ...) - integer-encode those columns, same as
    # the inline LabelEncoder step the old *_mushroom.py scripts used to
    # run before feeding the frame to BOSS/CPC/TM/CategoricalNB.
    def _to_int_codes(col):
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().all():
            # Re-index to a contiguous 0..k-1 range in sorted order - source
            # codes aren't guaranteed zero-based/contiguous (e.g. coil2000's
            # sociodemographic attributes run 1-43 with gaps), but
            # CategoricalNB indexes a feature's category array by the raw
            # integer value, so an ungapped code like 43 needs an array of
            # size 44 even though far fewer distinct values actually occur.
            # Dense-ranking preserves numeric order, unlike factorize below.
            return numeric.rank(method="dense").astype(int) - 1
        return pd.factorize(col.astype(str))[0]

    df = df.apply(_to_int_codes)

    # "Use whole samples" if the source has fewer rows than requested, rather
    # than erroring - same guard generate_regression.py's _prepare() applies.
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    return df.reset_index(drop=True)


def load_spect_dataset(sample_size=None, seed=42):
    """
    SPECT Heart (UCI, archive.ics.uci.edu/ml/machine-learning-databases/
    spect): 22 binary diagnostic attributes (F1-F22, from cardiac SPECT
    scans) + binary "Label" (normal/abnormal), 267 rows combined (80 train
    + 187 test on the original UCI split; concatenated here, same as the
    old experiments/run_tm_spect.py did). Uniform binary domain per
    column, like neticusdroid.csv/tuandromd.csv, but a much smaller
    dataset - useful as a fast categorical sanity check.

    Fetched over HTTP on first use and cached to a local CSV in this
    directory (mirrors _load_openml_frame's caching for the OpenML-backed
    loaders). Row-subsampled to `sample_size` if given (default: use
    every row, since the dataset is already small).

    Returns
    -------
    samples : pd.DataFrame
        All-categorical (integer-coded), row-subsampled to `sample_size`
        if given (or every row by default) - feed directly to a
        categorical-native algorithm/CI test (chi-square, BOSS's BDeu,
        etc.) or as the `samples` argument to a TM MB algorithm.
    """
    cache_path = DATA_DIR / "spect.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        col_names = ["label"] + [f"f{i}" for i in range(1, 23)]
        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect"
        train_df = pd.read_csv(f"{base_url}/SPECT.train", header=None, names=col_names)
        test_df = pd.read_csv(f"{base_url}/SPECT.test", header=None, names=col_names)
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        df.to_csv(cache_path, index=False)
    return _prepare_categorical(df, sample_size, seed)


def load_neticusdroid_dataset(sample_size=3000, seed=42):
    """
    NATICUSdroid Android-permissions malware dataset, from the local
    data/neticusdroid.csv already in this repo: 86 columns, all uniform
    binary (0/1) permission-usage flags, 29332 rows - unlike coil2000,
    every column here has the same 2-value domain. Row-subsampled to
    `sample_size` by default (or every row if the source has fewer) - same
    convention as the coil2000/generate_regression.py loaders, and fixes
    the stale absolute path (`/Users/kunaldumbre/Desktop/Bayesian_using_TM/
    data/neticusdroid.csv`) the older experiments_real_categorical/
    *_neticusdroid.py scripts had hardcoded inline.

    Returns
    -------
    Same shape as load_coil2000_dataset: a single all-categorical
    (integer-coded) DataFrame, row-subsampled to `sample_size`.
    """
    df = pd.read_csv(DATA_DIR / "neticusdroid.csv")
    return _prepare_categorical(df, sample_size, seed)


def load_mushroom_dataset(sample_size=3000, seed=42):
    """
    Mushroom (OpenML "mushroom", data_id=24): 22 attributes + "class"
    (edible/poisonous), real UCI Agaricus/Lepidota specimen records, 8124
    rows - domain sizes vary 1-12 per column ("veil-type" is constant at 1
    category; "gill-color" has 12). Row-subsampled to `sample_size` by
    default (or every row if the source has fewer). Replaces the older
    experiments_real_categorical/*_mushroom.py scripts' inline
    pd.read_csv(mushroom_url, ...) fetch - same source data, but cached
    locally and consistent with the other loaders here.

    Returns
    -------
    Same shape as load_coil2000_dataset: a single all-categorical
    (integer-coded) DataFrame, row-subsampled to `sample_size`.
    """
    df = _load_openml_frame("mushroom.csv", data_id=24)
    return _prepare_categorical(df, sample_size, seed)


def load_tuandromd_dataset(sample_size=3000, seed=42):
    """
    TUANDROMD (Tezpur University Android Malware Dataset), from the local
    data/tuandromd.csv already in this repo (restored from git history -
    the older experiments_real_categorical/*_tuandromd.py scripts had a
    stale absolute path, `/Users/kunaldumbre/Desktop/Bayesian_using_TM/
    data/tuandromd.csv`, that no longer resolves): 241 permission/API-call
    attributes + "Label" (malware/goodware), all uniform binary (0/1) -
    same domain shape as neticusdroid, different malware corpus. 4465 rows
    upstream, one row has a NaN label and is dropped by _prepare_categorical.
    Row-subsampled to `sample_size` by default (or every row if the source
    has fewer).

    Returns
    -------
    Same shape as load_coil2000_dataset: a single all-categorical
    (integer-coded) DataFrame, row-subsampled to `sample_size`.
    """
    df = pd.read_csv(DATA_DIR / "tuandromd.csv")
    return _prepare_categorical(df, sample_size, seed)
