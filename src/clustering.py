"""Explore development customers with a diagonal Gaussian mixture model.

Run from the project root: python -m src.clustering
"""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from . import config
from .experiment_preprocessing import make_feature_pipeline


def select_gmm(X_encoded, component_counts=range(2, 9), seed=config.RANDOM_SEED):
    """Choose the converged candidate with lowest BIC on the same feature space."""
    results = []
    best_model = None
    best_bic = np.inf
    for count in component_counts:
        model = GaussianMixture(
            n_components=count, covariance_type="diag", n_init=5,
            max_iter=500, reg_covar=1e-5, random_state=seed,
        ).fit(X_encoded)
        labels = model.predict(X_encoded)
        n_clusters = len(np.unique(labels))
        silhouette = np.nan
        if 1 < n_clusters < len(X_encoded):
            silhouette = silhouette_score(X_encoded, labels, metric="euclidean")
        bic = model.bic(X_encoded)
        results.append({
            "model": "GMM", "n_components": count, "n_clusters": n_clusters,
            "covariance_type": "diag", "n_features": X_encoded.shape[1],
            "rows": len(X_encoded), "seed": seed, "converged": model.converged_,
            "iterations": model.n_iter_, "bic": bic, "aic": model.aic(X_encoded),
            "silhouette": silhouette,
        })
        if model.converged_ and np.isfinite(silhouette) and bic < best_bic:
            best_model, best_bic = model, bic
    if best_model is None:
        raise ValueError("No converged GMM candidate has at least two occupied clusters.")
    selection = pd.DataFrame(results)
    selection["selected"] = selection["n_components"].eq(best_model.n_components)
    return best_model, selection


def evaluate_gmm(splits_dir=None):
    """Cluster verified development features and describe the resulting groups."""
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    manifest = json.loads((splits_dir / "manifest.json").read_bytes())
    content = (splits_dir / "development.csv").read_bytes()
    if sha256(content).hexdigest() != manifest["files_sha256"]["development.csv"]:
        raise ValueError("development.csv differs from the frozen split manifest.")
    development = pd.read_csv(BytesIO(content))
    X = development[manifest["features"]]
    X_encoded = make_feature_pipeline(X).fit_transform(X)
    seed = manifest["specification"]["seed"]
    model, selection = select_gmm(X_encoded, seed=seed)
    labels = model.predict(X_encoded)

    profiles = development.assign(cluster=labels).groupby("cluster").agg(
        customers=("Response", "size"),
        responders=("Response", "sum"),
        response_rate=("Response", "mean"),
        median_income=("Income", "median"),
        median_age=("Age", "median"),
        median_spending=("Total_Spent", "median"),
        median_purchases=("Total_Purchases", "median"),
        median_recency=("Recency", "median"),
        median_children=("Children", "median"),
    )
    profiles["customer_fraction"] = profiles["customers"] / len(development)
    profiles["response_lift"] = profiles["response_rate"] / development["Response"].mean()
    projection = PCA(n_components=2, svd_solver="full").fit_transform(X_encoded)
    visual = pd.DataFrame(projection, columns=["PC1", "PC2"]).assign(
        cluster=labels, Response=development["Response"].to_numpy(),
    )
    return selection, profiles.reset_index(), visual


if __name__ == "__main__":
    selection, profiles, _ = evaluate_gmm()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(config.TABLES_DIR / "gmm_model_selection.csv", index=False)
    profiles.to_csv(config.TABLES_DIR / "gmm_cluster_profiles.csv", index=False)
    print(selection.to_string(index=False))
    print(profiles.to_string(index=False))
