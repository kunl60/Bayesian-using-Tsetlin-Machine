from sklearn.feature_selection import mutual_info_classif
from algorithms.mb_bn_base import MBElbowBase


class MutualInfoMBElbow(MBElbowBase):

    def __init__(self, top_n, alpha=0.01, maxK=3, random_state=42):
        super().__init__(top_n=top_n, alpha=alpha, maxK=maxK)
        self.random_state = random_state

    def _score_candidates(self, label_encoded_samples, samples):
        columns = label_encoded_samples.columns.tolist()
        raw_scores = {}

        for target in columns:
            y = label_encoded_samples[target]
            if y.nunique() < 2:
                # Constant target column - nothing to score against it.
                continue

            feature_cols = [c for c in columns if c != target]
            X = label_encoded_samples[feature_cols]

            mi = mutual_info_classif(
                X, y,
                discrete_features=True,
                random_state=self.random_state,
            )
            raw_scores[target] = dict(zip(feature_cols, mi))

        return raw_scores
