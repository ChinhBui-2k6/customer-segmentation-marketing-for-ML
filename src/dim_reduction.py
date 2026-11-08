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
