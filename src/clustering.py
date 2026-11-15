"""K-Means experiments fitted on training data, without Response or metadata."""
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.pipeline import Pipeline

from src import config
from src.experiment_preprocessing import make_feature_pipeline


def cluster_metrics(X, labels):
    """Internal scores in the actual clustering space; undefined scores are NaN."""
    labels = np.asarray(labels)
    count = len(np.unique(labels))
    metrics = {"rows": len(labels), "clusters_present": count,
               "silhouette": np.nan, "davies_bouldin": np.nan,
               "calinski_harabasz": np.nan}
    if 1 < count < len(labels):
        metrics.update(silhouette=float(silhouette_score(X, labels)),
                       davies_bouldin=float(davies_bouldin_score(X, labels)),
                       calinski_harabasz=float(calinski_harabasz_score(X, labels)))
    return metrics


def elbow_candidate(candidates):
    """Heuristic: greatest drop below the line between normalized endpoints.

    Only an aid to reading the elbow plot, not a proof of an optimal K.
    """
    ordered = candidates.sort_values("k")
    if len(ordered) < 3:
        return None
    x, y = ordered.k.to_numpy(), ordered.inertia.to_numpy()
    if x[-1] == x[0] or y[0] <= y[-1]:
        return None
    normalized_x = (x - x[0]) / (x[-1] - x[0])
    normalized_y = (y - y[-1]) / (y[0] - y[-1])
    gap = 1 - normalized_x - normalized_y
    interior = int(np.argmax(gap[1:-1])) + 1
    return int(x[interior]) if gap[interior] > 1e-12 else None


def analyze_kmeans(X_train, X_validation, reduce=False, k_values=config.KMEANS_CANDIDATE_K,
                   seed=config.RANDOM_SEED, n_init=config.KMEANS_N_INIT):
    """Select K by maximum training silhouette; validation is evaluated once.

    Preprocessing/PCA is fitted once on this training partition. No labels are
    accepted. Both K selection and the elbow suggestion use training X only.
    Tie-breaking favors the smaller K. CSV scores are not comparable across
    feature spaces as if they measured the same geometry.
    """
    values = tuple(k_values)
    if (not values or len(set(values)) != len(values)
            or any(not isinstance(k, (int, np.integer)) or not 2 <= k < len(X_train) for k in values)):
        raise ValueError("Use distinct integer K values from 2 to n_train - 1.")
    features = make_feature_pipeline(X_train, reduce=reduce)
    train_space = features.fit_transform(X_train)
    validation_space = features.transform(X_validation)
    models, rows = {}, []
    for k in sorted(values):
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = model.fit_predict(train_space)
        metrics = cluster_metrics(train_space, labels)
        models[k] = model
        rows.append({"k": k, "inertia": float(model.inertia_),
                     "smallest_cluster": int(np.bincount(labels, minlength=k).min()),
                     **metrics})
    candidates = pd.DataFrame(rows)
    valid = candidates.dropna(subset=["silhouette"])
    if valid.empty:
        raise ValueError("No candidate produced at least two nonempty clusters.")
    selected_k = int(valid.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    model = models[selected_k]
    train_labels = model.labels_.copy()
    validation_labels = model.predict(validation_space)
    fitted_pipeline = Pipeline([*features.steps, ("kmeans", model)])
    evaluation = []
    for role, X, labels in [("train", train_space, train_labels),
                             ("validation", validation_space, validation_labels)]:
        distances = model.transform(X)
        evaluation.append({"partition": role, "selected_k": selected_k,
                           "mean_squared_distance_to_centroid": float(np.mean(np.min(distances, axis=1) ** 2)),
                           **cluster_metrics(X, labels)})
    return dict(feature_pipeline=features, pipeline=fitted_pipeline, model=model,
                train_space=train_space, validation_space=validation_space,
                train_labels=train_labels, validation_labels=validation_labels,
                selected_k=selected_k, elbow_k=elbow_candidate(candidates),
                candidates=candidates, evaluation=pd.DataFrame(evaluation))


def initialization_stability(X, reference_labels, k, seeds=config.KMEANS_STABILITY_SEEDS,
                             n_init=config.KMEANS_N_INIT):
    """ARI across initialization seeds, conditional on a fixed training sample."""
    rows = []
    for seed in seeds:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        rows.append({"seed": seed, "inertia": model.inertia_,
                     "ari_vs_reference": adjusted_rand_score(reference_labels, model.labels_)})
    return pd.DataFrame(rows)


def describe_clusters(clean_features, response, labels):
    """Describe original-unit customer features; Response is post-hoc only.

    Inputs use positional alignment. Metadata must never be part of features.
    """
    # The shared factory enforces the same feature exclusion rules.
    make_feature_pipeline(clean_features)
    if len(clean_features) != len(response) or len(response) != len(labels):
        raise ValueError("Features, response and labels must have matching rows.")
    response = np.asarray(response)
    if not np.isin(response, [0, 1]).all():
        raise ValueError("Response must contain only 0 and 1.")
    frame = clean_features.reset_index(drop=True).copy()
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame if c not in numeric]
    frame["cluster"] = np.asarray(labels)
    frame["Response"] = response
    grouped = frame.groupby("cluster")
    summary = grouped.Response.agg(rows="size", response_1="sum", response_rate="mean")
    summary["fraction"] = summary.rows / len(frame)
    summary["response_lift"] = summary.response_rate / response.mean() if response.mean() else np.nan
    categories = []
    for cluster, part in grouped:
        for column in categorical:
            for value, count in part[column].value_counts(dropna=False).items():
                categories.append({"cluster": cluster, "feature": column, "value": value,
                                   "rows": int(count), "fraction": count / len(part)})
    return dict(summary=summary, numeric_mean=grouped[numeric].mean(),
                numeric_median=grouped[numeric].median(),
                categories=pd.DataFrame(categories))


if __name__ == "__main__":
    notebook = Path(__file__).resolve().parent.parent / "notebooks" / "03_clustering.ipynb"
    print("This module provides clustering functions. Run the notebook for the full experiment:")
    print(f'jupyter notebook "{notebook}"')
