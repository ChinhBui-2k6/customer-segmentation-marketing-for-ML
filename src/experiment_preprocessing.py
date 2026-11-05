"""Train-only preprocessing factories, also safe to clone inside group-aware CV."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, validate_data

from src import preprocessing


class OneThirdPCA(TransformerMixin, BaseEstimator):
    """Learn floor(encoded feature count / 3) PCs on the current training fold."""

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=float)
        self.n_components_ = max(1, X.shape[1] // 3)
        if self.n_components_ > X.shape[0]:
            raise ValueError("Too few training rows for one-third PCA.")
        self.pca_ = PCA(n_components=self.n_components_, svd_solver="full")
        self.pca_.fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        return self.pca_.transform(validate_data(self, X, reset=False, dtype=float))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "pca_")
        return np.array([f"PC{i + 1}" for i in range(self.n_components_)], dtype=object)


def make_feature_pipeline(X_train, reduce=False):
    """Return an unfitted pipeline; schema comes from training columns only.

    Append the classifier/regressor and fit the whole Pipeline inside each CV
    fold. Exported transformed CSVs are for fixed train/validation experiments,
    not input for hyperparameter search across new folds.
    """
    forbidden = {"Response", "ID", "customer_id", "source_row", "processed_row",
                 "duplicate_group", "partition", "development_fold",
                 "Z_CostContact", "Z_Revenue"}
    if forbidden.intersection(X_train.columns):
        raise ValueError("Remove target, metadata and excluded columns from X.")
    numeric, categorical = preprocessing.split_feature_types(X_train)
    if set(numeric + categorical) != set(X_train.columns):
        raise ValueError("Unsupported feature dtype.")
    steps = [("preprocessor", preprocessing.build_preprocessor(numeric, categorical))]
    if reduce:
        steps.append(("pca", OneThirdPCA()))
    return Pipeline(steps)
