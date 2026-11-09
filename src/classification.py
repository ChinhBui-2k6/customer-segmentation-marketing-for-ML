"""Evaluate Logistic Regression and export the scores shared by the regressors.

Run after the frozen splits are generated: python -m src.classification
"""
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

from . import config
from .experiment_preprocessing import make_feature_pipeline


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
    print(export_logistic_handoff())
