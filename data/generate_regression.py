"""
Real-world regression datasets, all with 100% continuous (numeric) columns -
unlike generate_ecoli/generate_sachs, these have no known ground-truth DAG,
so they're for Markov-blanket/skeleton discovery and feature evaluation only
(see experiments_real_regression/), not precision/recall against a true graph.

Loaders download via sklearn's fetch_openml on first use (Gas Turbine is the
one exception - not on OpenML, fetched directly from its UCI zip instead).
The raw frame is cached to a local CSV in this directory (mirrors
data/neticusdroid.csv) so repeated runs across the many per-algorithm
scripts don't re-hit the network.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).parent

GAS_TURBINE_URL = (
    "https://archive.ics.uci.edu/static/public/551/"
    "gas+turbine+co+and+nox+emission+data+set.zip"
)


def _load_openml_frame(data_id, cache_file):
    cache_path = DATA_DIR / cache_file
    if cache_path.exists():
        return pd.read_csv(cache_path)

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    df.columns = [str(c) for c in df.columns]
    df.to_csv(cache_path, index=False)
    return df


def _load_uci_zip_frame(url, cache_file):
    # Gas Turbine isn't on OpenML, so this fetches its UCI zip directly
    # instead of going through _load_openml_frame - it ships as one CSV per
    # year (gt_2011.csv..gt_2015.csv, same schema), concatenated here into
    # a single frame. Caches the same way _load_openml_frame does.
    cache_path = DATA_DIR / cache_file
    if cache_path.exists():
        return pd.read_csv(cache_path)

    with urllib.request.urlopen(url) as resp:
        raw_zip = resp.read()
    archive = zipfile.ZipFile(io.BytesIO(raw_zip))
    df = pd.concat(
        (pd.read_csv(io.BytesIO(archive.read(name))) for name in sorted(archive.namelist())),
        ignore_index=True,
    )
    df.to_csv(cache_path, index=False)
    return df


def _prepare(df, sample_size, n_bins, seed):
    # Strip underscores from column names - same fix generate_sachs.py applies
    # to node names. wtm_nl_regressor.py/wtm_bn_regressor.py parse
    # one-hot column names like "total_rooms_bin3" by splitting on "_", so an
    # underscore inside the *original* column name (e.g. "total_rooms" and
    # "total_bedrooms" both starting with "total_") causes it to misparse
    # which raw column a literal belongs to.
    df = df.rename(columns=lambda c: str(c).replace("_", ""))

    # All-numeric already for the datasets below, but coerce/drop defensively
    # in case OpenML serves a stray non-numeric/missing cell.
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
    raw_samples = df.reset_index(drop=True)

    # Quantile-bin each column independently. Skewed/clustered real-world
    # columns (unlike ecoli70/sachs's synthetic ones) often collapse some
    # quantile edges together, so a column can end up with fewer than
    # n_bins actual bins - a fixed n_bins-length label list would then
    # mismatch pd.qcut's edge count. Let qcut assign integer codes instead
    # and label from those, so each column gets exactly as many "binI"
    # labels as it actually has distinct bins.
    def _qbin(col):
        codes = pd.qcut(col, q=n_bins, duplicates="drop").cat.codes
        return codes.map(lambda i: f"bin{i}")

    categorical_samples = raw_samples.apply(_qbin).astype(str)

    return raw_samples, categorical_samples


def load_california_housing_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    California district-level housing data (OpenML "houses", data_id=537):
    8 continuous predictors + continuous target (median_house_value),
    20,640 rows.

    Returns
    -------
    raw_samples : pd.DataFrame
        Continuous samples (optionally row-subsampled to `sample_size`) -
        feed directly to a continuous-native algorithm (CPCAlgorithm
        (tester="fisherz"), BOSS/DGES/DAGMA).
    categorical_samples : pd.DataFrame
        Each column quantile-binned into `n_bins` categories (string labels)
        - the Boolean-literal-ready form needed as the `samples` argument to
        the TM MB (Con) Regressor algorithms.
    """
    df = _load_openml_frame(537, "california_housing.csv")
    return _prepare(df, sample_size, n_bins, seed)


def load_superconductivity_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    Superconductivity critical-temperature data (OpenML data_id=43174):
    81 continuous material-property features + continuous target
    (critical_temp), 21,263 rows. Row-subsampled by default (81 features
    already makes CI-test-heavy algorithms slow) - pass sample_size=None
    for the full dataset.

    Returns
    -------
    Same shape as load_california_housing_dataset: (raw_samples, categorical_samples).
    """
    df = _load_openml_frame(43174, "superconductivity.csv")
    return _prepare(df, sample_size, n_bins, seed)


def load_year_prediction_msd_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    YearPredictionMSD - a subset of the Million Song Dataset (OpenML
    "Year_Prediction_MSD", data_id=46672): 90 continuous audio-timbre
    features (12 timbre averages + 78 timbre covariances) + continuous
    target (year the track was released), 515,345 rows. By far the largest
    dataset here, so it's row-subsampled by default like superconductivity
    - pass sample_size=None for the full dataset (fetch_openml's first
    download is then correspondingly large, ~500MB).

    Returns
    -------
    Same shape as load_california_housing_dataset: (raw_samples, categorical_samples).
    """
    df = _load_openml_frame(46672, "year_prediction_msd.csv")
    return _prepare(df, sample_size, n_bins, seed)


def load_musk_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    Musk (Version 2) - molecule conformation data (OpenML data_id=46615):
    166 continuous molecule-shape descriptors ("f1".."f166"), 6598 rows.
    The original "class" column (musk vs. non-musk) is a binary
    classification label, not a continuous feature, so it's dropped here -
    every loader in this module returns 100% continuous columns. A
    different domain (chemistry/molecular shape) from the other loaders
    here. Row-subsampled by default like superconductivity - 166 total
    columns already makes CI-test-heavy algorithms slow - pass
    sample_size=None for the full dataset.

    Returns
    -------
    Same shape as load_california_housing_dataset: (raw_samples, categorical_samples).
    """
    df = _load_openml_frame(46615, "musk.csv").drop(columns=["class"])
    return _prepare(df, sample_size, n_bins, seed)


def load_ecg5000_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    ECG5000 - heartbeat waveform data (OpenML data_id=44793): 140
    continuous amplitude readings along a single heartbeat ("col_0"..
    "col_139"), 4998 rows. The original "col_140" column is a binary
    classification label (heartbeat type), not a continuous feature, so
    it's dropped here - every loader in this module returns 100%
    continuous columns. Biomedical/signal domain, distinct from the other
    loaders here. Row-subsampled by default like superconductivity - 140
    total columns already makes CI-test-heavy algorithms slow - pass
    sample_size=None for the full dataset.

    Returns
    -------
    Same shape as load_california_housing_dataset: (raw_samples, categorical_samples).
    """
    df = _load_openml_frame(44793, "ecg5000.csv").drop(columns=["col_140"])
    return _prepare(df, sample_size, n_bins, seed)


def load_parkinsons_telemonitoring_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    Parkinsons Telemonitoring - biomedical voice/speech-signal data (OpenML
    data_id=4531): 16 continuous voice-signal measures (Jitter, Shimmer,
    NHR, HNR, RPDE, DFA, PPE, etc.) plus age and test_time, and 2 continuous
    targets (motor_UPDRS, total_UPDRS - Parkinson's symptom severity
    scores), 5,875 voice recordings from 42 patients. The original
    "subject." (patient id) and "sex" columns are an identifier and a
    binary/categorical field, not continuous features, so they're dropped
    here - every loader in this module returns 100% continuous columns.
    Speech/acoustic-signal domain, distinct from the other loaders here.
    Row-subsampled by default like superconductivity.

    Returns
    -------
    Same shape as load_california_housing_dataset: (raw_samples, categorical_samples).
    """
    df = _load_openml_frame(4531, "parkinsons_telemonitoring.csv")
    df = df.drop(columns=["subject.", "sex"])
    return _prepare(df, sample_size, n_bins, seed)


def load_gas_turbine_emission_dataset(sample_size=3000, n_bins=5, seed=42):
    """
    Gas Turbine CO and NOx Emission - industrial sensor-signal data (UCI
    dataset 551; not on OpenML, so fetched directly via _load_uci_zip_frame):
    9 continuous sensor readings (ambient temperature/pressure/humidity,
    turbine inlet/exhaust temperature, compressor discharge pressure, etc.)
    plus 2 continuous targets (CO, NOx emissions), 36,733 hourly readings
    from a turbine in Turkey spanning 2011-2015 (one CSV per year,
    concatenated). Sensor-signal domain, distinct from the other loaders
    here. Row-subsampled by default like superconductivity.

    Returns
    -------
    Same shape as load_california_housing_dataset: (raw_samples, categorical_samples).
    """
    df = _load_uci_zip_frame(GAS_TURBINE_URL, "gas_turbine_emission.csv")
    return _prepare(df, sample_size, n_bins, seed)
