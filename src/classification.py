"""Gaussian NB and RBF SVM with group-aware, train-only model selection."""
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.evaluation import classification_metrics, positive_scores
from src.experiment_preprocessing import make_feature_pipeline


class FoldBalancedGaussianNB(GaussianNB):
    """Calculate balanced weights inside fit, on the current training fold only.

    GridSearchCV clones this estimator for each fold. Passing a single set of
    full-training weights into all folds would use their held-out frequencies.
    """
    def __init__(self, *, var_smoothing=1e-9, balanced=False):
        super().__init__(var_smoothing=var_smoothing)
        self.balanced = balanced

    def fit(self, X, y, sample_weight=None):
        if self.balanced:
            weights = compute_sample_weight("balanced", y)
            sample_weight = weights if sample_weight is None else weights * sample_weight
        return super().fit(X, y, sample_weight=sample_weight)


def make_classifier(X_train, model, reduce=False, balanced=False):
    """Fresh pipeline: preprocessing, optional fold-specific PCA, classifier."""
    features = make_feature_pipeline(X_train, reduce=reduce)
    if model == "gaussian_nb":
        classifier = FoldBalancedGaussianNB(balanced=balanced)
    elif model == "svm_rbf":
        classifier = SVC(kernel="rbf", C=1.0, gamma="scale",
                         class_weight="balanced" if balanced else None)
    else:
        raise ValueError("model must be gaussian_nb or svm_rbf.")
    return Pipeline([*features.steps, ("classifier", classifier)])


def grouped_training_cv(X_train, y_train, groups, n_splits=config.CLASSIFICATION_CV_FOLDS):
    """Materialize audited train-only CV splits, retaining duplicate groups."""
    y, group_ids = np.asarray(y_train), np.asarray(groups)
    if len(X_train) != len(y) or len(y) != len(group_ids) or pd.isna(group_ids).any():
        raise ValueError("X, labels and nonmissing groups must have matching rows.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training requires both classes 0 and 1.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    folds = list(splitter.split(X_train, y, group_ids))
    for train, validation in folds:
        if set(group_ids[train]) & set(group_ids[validation]):
            raise ValueError("Duplicate group overlaps a CV fold.")
        if any(set(np.unique(y[part])) != {0, 1} for part in (train, validation)):
            raise ValueError("A CV partition is missing a class; use fewer folds.")
    return folds


def tune_classifier(X_train, y_train, groups, model, reduce=False,
                    n_splits=config.CLASSIFICATION_CV_FOLDS, param_grid=None):
    """Optimize mean group-CV AP on train. No outer validation input accepted.

    All preprocessing, PCA and balancing refit inside every training fold.
    AP is selected for ranking under class imbalance; trapezoidal PR-AUC is
    reported separately in evaluation.py. Defaults do not tune thresholds.
    """
    pipeline = make_classifier(X_train, model, reduce=reduce)
    folds = grouped_training_cv(X_train, y_train, groups, n_splits=n_splits)
    if param_grid is None:
        if model == "gaussian_nb":
            param_grid = {"classifier__var_smoothing": config.NB_SMOOTHING_GRID,
                          "classifier__balanced": [False, True]}
        else:
            param_grid = {"classifier__C": config.SVM_C_GRID,
                          "classifier__gamma": config.SVM_GAMMA_GRID,
                          "classifier__class_weight": [None, "balanced"]}
    search = GridSearchCV(pipeline, param_grid=param_grid,
                          scoring={"average_precision": "average_precision", "f1": "f1",
                                   "recall": "recall", "roc_auc": "roc_auc"},
                          refit="average_precision", cv=folds, n_jobs=1,
                          return_train_score=True, error_score="raise")
    search.fit(X_train, y_train)
    return search


def evaluate_classifier(estimator, X, y):
    predictions = estimator.predict(X)
    scores = positive_scores(estimator, X)
    return classification_metrics(y, predictions, scores), predictions, scores


# Tuananh: imported from origin/main at 191d94c.
from hashlib import sha256

from io import BytesIO

import json

from pathlib import Path

from tempfile import TemporaryDirectory

import joblib

import pandas as pd

import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

from src import config

from src.experiment_preprocessing import make_feature_pipeline

def fit_logistic_source(X_train, y_train, groups, X_validation, seed):
    """Fit the baseline on full features and score train out of fold."""
    pipeline = make_feature_pipeline(X_train)
    pipeline.steps.append(("logistic", LogisticRegression(max_iter=1000, random_state=seed)))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_scores = cross_val_predict(
        pipeline, X_train, y_train, groups=groups, cv=cv, method="predict_proba"
    )[:, 1]
    pipeline.fit(X_train, y_train)
    validation_scores = pipeline.predict_proba(X_validation)[:, 1]
    return pipeline, train_scores, validation_scores

def _score_table(metadata, labels, probabilities):
    return metadata[["processed_row", "source_row"]].assign(
        Response=labels.to_numpy(), probability_response_1=probabilities
    )

def _summarize(split, role, labels, scores):
    predicted = scores >= 0.5
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "split": split,
        "partition": role,
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "pr_auc": auc(recall, precision),
        "roc_auc": roc_auc_score(labels, scores),
    }

def _iter_verified_splits(splits_dir, manifest):
    def read_verified_csv(name, **kwargs):
        content = (splits_dir / name).read_bytes()
        if sha256(content).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError(f"Split file differs from its manifest: {name}")
        return pd.read_csv(BytesIO(content), **kwargs)

    assignments = read_verified_csv("assignments.csv", dtype={"duplicate_group": str})
    development = assignments.loc[assignments["partition"].eq("development")]
    for fraction in manifest["specification"]["validation_sizes"]:
        percentage = round(fraction * 100)
        split = f"{100 - percentage}_{percentage}"
        validation_mask = development["development_fold"].lt(round(fraction * 10))
        train_rows = development.loc[~validation_mask]
        validation_rows = development.loc[validation_mask]
        train = read_verified_csv(f"train_{split}.csv")
        validation = read_verified_csv(f"validation_{split}.csv")
        if len(train) != len(train_rows) or len(validation) != len(validation_rows):
            raise ValueError(f"Row count differs from split assignments: {split}")
        if set(train_rows["duplicate_group"]) & set(validation_rows["duplicate_group"]):
            raise ValueError(f"Duplicate groups overlap: {split}")
        yield split, train, validation, train_rows, validation_rows

def evaluate_logistic(splits_dir=None):
    """Tune on train groups, then score train and validation for every scenario."""
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    manifest = json.loads((splits_dir / "manifest.json").read_bytes())
    features = manifest["features"]
    target = manifest["target"]
    seed = manifest["specification"]["seed"]

    # scikit-learn 1.8 replaced the penalty argument with l1_ratio.
    version = tuple(map(int, sklearn.__version__.split(".")[:2]))
    penalty_grid = (
        {"logistic__l1_ratio": [0.0, 1.0]} if version >= (1, 8)
        else {"logistic__penalty": ["l2", "l1"]}
    )
    parameters = {"logistic__C": [0.01, 0.1, 1.0, 10.0], **penalty_grid}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    metrics, validation_predictions = [], []
    for split, train, validation, train_rows, _ in _iter_verified_splits(splits_dir, manifest):
        X_train, X_validation = train[features], validation[features]
        y_train, y_validation = train[target], validation[target]
        groups = train_rows["duplicate_group"]
        for representation, reduce in (("full", False), ("pca_one_third", True)):
            for weight in (None, "balanced"):
                pipeline = make_feature_pipeline(X_train, reduce=reduce)
                pipeline.steps.append(("logistic", LogisticRegression(
                    solver="liblinear", class_weight=weight, max_iter=1000, random_state=seed
                )))
                search = GridSearchCV(
                    pipeline, parameters, cv=cv, scoring="average_precision", error_score="raise"
                )
                search.fit(X_train, y_train, groups=groups)
                model = search.best_estimator_
                logistic = model.named_steps["logistic"]
                penalty = (
                    ("l1" if logistic.l1_ratio == 1 else "l2") if version >= (1, 8)
                    else logistic.penalty
                )
                setting = {
                    "model": "Logistic Regression",
                    "representation": representation,
                    "n_features": logistic.n_features_in_,
                    "class_weight": weight or "default",
                    "seed": seed,
                    "penalty": penalty,
                    "C": logistic.C,
                    "cv_average_precision": search.best_score_,
                }
                for role, X, labels in (
                    ("train", X_train, y_train), ("validation", X_validation, y_validation)
                ):
                    probabilities = model.predict_proba(X)[:, 1]
                    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= 0.5, labels=[0, 1]).ravel()
                    metrics.append({
                        **setting, **_summarize(split, role, labels, probabilities),
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    })
                    if role == "validation":
                        validation_predictions.append(pd.DataFrame({
                            "split": split, "representation": representation,
                            "class_weight": setting["class_weight"],
                            "Response": labels, "probability_response_1": probabilities,
                        }))
    return pd.DataFrame(metrics), pd.concat(validation_predictions, ignore_index=True)

def export_logistic_handoff(splits_dir=None, output_dir=None):
    """Publish all three handoff splits together in a new directory."""
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    output_dir = (
        Path(output_dir) if output_dir is not None
        else config.PROCESSED_DATA_DIR / "logistic_scores"
    )
    if output_dir.exists():
        raise FileExistsError(f"Logistic handoff already exists: {output_dir}")

    manifest_bytes = (splits_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    features = manifest["features"]
    seed = manifest["specification"]["seed"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as staging_root:
        staging = Path(staging_root) / "logistic_scores"
        staging.mkdir()
        metrics = []
        for split, train, validation, train_rows, validation_rows in _iter_verified_splits(splits_dir, manifest):
            y_train, y_validation = train["Response"], validation["Response"]
            model, train_scores, validation_scores = fit_logistic_source(
                train[features], y_train, train_rows["duplicate_group"], validation[features], seed
            )
            directory = staging / split
            directory.mkdir()
            _score_table(train_rows, y_train, train_scores).to_csv(
                directory / "train_oof_scores.csv", index=False
            )
            _score_table(validation_rows, y_validation, validation_scores).to_csv(
                directory / "validation_scores.csv", index=False
            )
            joblib.dump(model, directory / "source_model.joblib")
            metrics.extend([
                _summarize(split, "train_oof", y_train, train_scores),
                _summarize(split, "validation", y_validation, validation_scores),
            ])

        pd.DataFrame(metrics).to_csv(staging / "baseline_metrics.csv", index=False)
        metadata = {
            "split_manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "sklearn_version": sklearn.__version__,
            "seed": seed,
            "train_score_method": "5-fold stratified group out-of-fold predict_proba",
            "validation_score_method": "predict_proba from the pipeline fit on full train",
            "positive_label": 1,
            "feature_branch": "full",
        }
        (staging / "manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return output_dir


if __name__ == "__main__":
    notebook = Path(__file__).resolve().parent.parent / "notebooks" / "04_classification.ipynb"
    if "--export-logistic-scores" in sys.argv:
        print(export_logistic_handoff())
    else:
        print("Open the notebook and select Restart Kernel and Run All:")
        print(f'jupyter notebook "{notebook}"')
