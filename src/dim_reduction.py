"""PCA diagnostics fitted exclusively on a training partition.

Kernel PCA will be added by Tuananh. Components refer to the encoded/scaled
feature space, not the raw mixed-unit customer data.
"""
from pathlib import Path
import sys

# Support `python dim_reduction.py` from src as well as package imports.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.experiment_preprocessing import make_feature_pipeline


def component_statistics(scores):
    """Sample std (ddof=1), mean and pandas unbiased skewness per component."""
    return pd.DataFrame({"mean": scores.mean(), "std": scores.std(ddof=1),
                         "skewness": scores.skew(), "min": scores.min(),
                         "median": scores.median(), "max": scores.max()}).rename_axis("component")


def analyze_pca(X_train, X_validation):
    """Fit preprocessing and full PCA on train; transform validation only.

    Full PCA supports the scree/cumulative variance report. The first k scores
    are the one-third-dimensional representation for downstream experiments.
    No labels or metadata are accepted by the preprocessing factory.
    """
    preprocessing = make_feature_pipeline(X_train)
    encoded_train = preprocessing.fit_transform(X_train)
    encoded_validation = preprocessing.transform(X_validation)
    pca = PCA(svd_solver="full").fit(encoded_train)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    train_scores = pd.DataFrame(pca.transform(encoded_train), columns=names, index=X_train.index)
    validation_scores = pd.DataFrame(pca.transform(encoded_validation), columns=names, index=X_validation.index)
    k = max(1, encoded_train.shape[1] // 3)
    if k > pca.n_components_:
        raise ValueError("Too few training rows for one-third PCA.")
    variance = pd.DataFrame({"component": names, "eigenvalue": pca.explained_variance_,
                             "explained_variance_ratio": pca.explained_variance_ratio_,
                             "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_)})
    weights = pd.DataFrame(pca.components_.T,
                           index=preprocessing.get_feature_names_out(), columns=names)
    weights.index.name = "encoded_feature"
    summary = {"train_rows": len(X_train), "validation_rows": len(X_validation),
               "encoded_features": encoded_train.shape[1], "retained_components": k,
               "retained_variance_ratio": float(pca.explained_variance_ratio_[:k].sum())}
    for threshold in (.90, .95):
        summary[f"components_for_{int(threshold * 100)}pct"] = int(
            np.searchsorted(variance.cumulative_variance_ratio.to_numpy(), threshold) + 1)
    return dict(preprocessing=preprocessing, pca=pca, train_scores=train_scores,
                validation_scores=validation_scores, variance=variance, weights=weights,
                train_statistics=component_statistics(train_scores),
                validation_statistics=component_statistics(validation_scores), summary=summary)


# Tuananh: imported from origin/main at 191d94c.
import numpy as np

import pandas as pd

import math

from typing import Tuple, Optional

from sklearn.decomposition import KernelPCA, PCA

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.pipeline import Pipeline

from sklearn.utils.validation import check_is_fitted

from scipy.stats import skew

import warnings

def fit_kernel_pca_rbf(
    X_train: np.ndarray, 
    n_components: Optional[int] = None, 
    gamma: Optional[float] = None
) -> KernelPCA:
    """
    Huấn luyện Kernel PCA với kernel RBF.
    """
    kpca = KernelPCA(
        n_components=n_components,
        kernel='rbf',
        gamma=gamma,
        random_state=42
    )
    kpca.fit(X_train)
    return kpca

def compute_kernel_explained_variance(
    kpca_model: KernelPCA,
    n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính phương sai thành phần và tỷ lệ so với toàn bộ kernel RBF."""
    check_is_fitted(kpca_model, ("eigenvalues_", "X_fit_", "gamma_"))
    if kpca_model.kernel != "rbf":
        raise ValueError("Hàm tính phương sai này yêu cầu kernel RBF.")
    if n_samples < 2 or n_samples != kpca_model.X_fit_.shape[0]:
        raise ValueError("n_samples phải khớp số mẫu fit và có ít nhất hai mẫu.")

    eigenvalues = kpca_model.eigenvalues_
    if np.any(eigenvalues < 0):
        warnings.warn("Phát hiện trị riêng âm do sai số số học. Đã cắt về 0.")
        eigenvalues = np.maximum(eigenvalues, 0.0)

    kernel = rbf_kernel(kpca_model.X_fit_, gamma=kpca_model.gamma_)
    # trace(H K H) = trace(K) - sum(K) / n với H là ma trận định tâm.
    centered_trace = np.trace(kernel) - kernel.sum() / n_samples
    explained_variance = eigenvalues / (n_samples - 1)
    explained_variance_ratio = (
        eigenvalues / centered_trace if centered_trace > 0 else np.zeros_like(eigenvalues)
    )
    return explained_variance, explained_variance_ratio

def create_shared_pca_pipeline(
    d_features: int, 
    standard_pca_instance: PCA
) -> Pipeline:
    """
    Tạo Pipeline PCA dùng chung giảm còn 1/3 số chiều: k = max(1, floor(d/3)).
    """
    if d_features <= 0:
        raise ValueError("Số lượng đặc trưng d_features phải lớn hơn 0.")
        
    k_components = max(1, math.floor(d_features / 3))
    print(f"[INFO] Số chiều gốc (d): {d_features} -> Số chiều sau PCA (k): {k_components}")
    
    standard_pca_instance.set_params(n_components=k_components, random_state=42)
    
    pipeline = Pipeline(steps=[
        ('pca_shared', standard_pca_instance)
    ])
    
    return pipeline

def generate_components_statistics(
    X_transformed: np.ndarray, 
    n_components_to_show: int = 6
) -> pd.DataFrame:
    """
    Lập bảng thống kê mean, std, skewness cho các thành phần.
    """
    n_cols = min(n_components_to_show, X_transformed.shape[1])
    df_transformed = pd.DataFrame(
        X_transformed[:, :n_cols], 
        columns=[f"Component_{i+1}" for i in range(n_cols)]
    )
    
    stats_dict = {
        "Mean": df_transformed.mean(),
        "Std": df_transformed.std(),
        "Skewness": df_transformed.apply(skew)
    }
    
    return pd.DataFrame(stats_dict)


if __name__ == "__main__":
    notebook = Path(__file__).resolve().parent.parent / "notebooks" / "02_dimensionality_reduction.ipynb"
    print("This module provides PCA functions; no analysis has been run.")
    print("For the complete analysis, open the notebook and select Restart Kernel and Run All:")
    print(f'jupyter notebook "{notebook}"')
