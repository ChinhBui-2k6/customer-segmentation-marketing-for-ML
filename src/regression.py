"""Evaluate Linear Regression against the shared Logistic Regression scores.

Run after the Logistic score handoff exists: python -m src.regression
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from . import config
from .classification import _iter_verified_splits
from .experiment_preprocessing import make_feature_pipeline


def _read_aligned_scores(path, metadata, labels):
    """Join source scores to verified split rows and reject mismatched records."""
    scores = pd.read_csv(path).set_index("processed_row")
    if not scores.index.is_unique or len(scores) != len(metadata) or set(scores.index) != set(metadata["processed_row"]):
        raise ValueError(f"Score rows differ from the split assignments: {path}")
    scores = scores.loc[metadata["processed_row"]]
    if not np.array_equal(scores["source_row"], metadata["source_row"]):
        raise ValueError(f"Source row identifiers differ from the split assignments: {path}")
    if not np.array_equal(scores["Response"], labels.to_numpy()):
        raise ValueError(f"Response labels differ from the split: {path}")
    probabilities = scores["probability_response_1"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(f"Logistic source scores must be finite and between zero and one: {path}")
    return probabilities


def evaluate_linear_regression(splits_dir=None, scores_dir=None):
    """Fit on train-only features and compare full and one-third PCA inputs."""
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    scores_dir = (
        Path(scores_dir) if scores_dir is not None
        else config.PROCESSED_DATA_DIR / "logistic_scores"
    )
    manifest_bytes = (splits_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    source_manifest = json.loads((scores_dir / "manifest.json").read_bytes())
    if source_manifest["split_manifest_sha256"] != sha256(manifest_bytes).hexdigest():
        raise ValueError("Logistic source scores refer to a different split manifest.")
    seed = manifest["specification"]["seed"]
    if source_manifest["positive_label"] != 1 or source_manifest["seed"] != seed:
        raise ValueError("Logistic source label or seed differs from the experiment.")

    features = manifest["features"]
    target = manifest["target"]
    metrics, validation_predictions = [], []
    for split, train, validation, train_rows, validation_rows in _iter_verified_splits(splits_dir, manifest):
        y_train = _read_aligned_scores(
            scores_dir / split / "train_oof_scores.csv", train_rows, train[target]
        )
        y_validation = _read_aligned_scores(
            scores_dir / split / "validation_scores.csv", validation_rows, validation[target]
        )
        X_train, X_validation = train[features], validation[features]
        for representation, reduce in (("full", False), ("pca_one_third", True)):
            pipeline = make_feature_pipeline(X_train, reduce=reduce)
            pipeline.steps.append(("linear", LinearRegression()))
            pipeline.fit(X_train, y_train)
            for role, X, scores in (
                ("train", X_train, y_train), ("validation", X_validation, y_validation)
            ):
                predicted = pipeline.predict(X)
                metrics.append({
                    "model": "Linear Regression",
                    "split": split,
                    "representation": representation,
                    "partition": role,
                    "rows": len(scores),
                    "n_features": pipeline.named_steps["linear"].n_features_in_,
                    "seed": seed,
                    "rmse": root_mean_squared_error(scores, predicted),
                    "mae": mean_absolute_error(scores, predicted),
                    "r2": r2_score(scores, predicted),
                    "predictions_outside_unit_interval": int(((predicted < 0) | (predicted > 1)).sum()),
                })
                if role == "validation":
                    validation_predictions.append(pd.DataFrame({
                        "split": split,
                        "representation": representation,
                        "processed_row": validation_rows["processed_row"].to_numpy(),
                        "source_probability": scores,
                        "prediction": predicted,
                    }))
    return pd.DataFrame(metrics), pd.concat(validation_predictions, ignore_index=True)


if __name__ == "__main__":
    metrics, _ = evaluate_linear_regression()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.TABLES_DIR / "linear_regression_metrics.csv", index=False)
    print(metrics.to_string(index=False))
