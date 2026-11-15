"""PCA mathematical checks and holdout independence."""
import unittest
import numpy as np
import pandas as pd
from src.dim_reduction import analyze_pca
from src.dim_reduction import fit_kernel_pca_rbf, compute_kernel_explained_variance
from sklearn.metrics.pairwise import rbf_kernel
from src.experiment_preprocessing import make_feature_pipeline


class PCATests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(60, 8)), columns=list('abcdefgh'))
        frame['category'] = ['A', 'B'] * 30
        frame.loc[0, 'a'] = np.nan
        return frame.iloc[:45].copy(), frame.iloc[45:].copy()

    def test_covariance_and_reconstruction(self):
        train, val = self.data()
        result = analyze_pca(train, val)
        scores = result['train_scores']
        np.testing.assert_allclose(scores.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(scores.cov(), np.diag(result['pca'].explained_variance_), atol=1e-12)
        encoded = result['preprocessing'].transform(train)
        np.testing.assert_allclose(result['pca'].inverse_transform(scores), encoded, atol=1e-12)
        self.assertAlmostEqual(result['variance'].explained_variance_ratio.sum(), 1)

    def test_extreme_validation_does_not_change_training_fit(self):
        train, val = self.data()
        first = analyze_pca(train, val)
        changed = val.copy()
        changed.loc[:, list('abcdefgh')] = 1e8
        changed['category'] = 'NEW'
        second = analyze_pca(train, changed)
        np.testing.assert_array_equal(first['pca'].components_, second['pca'].components_)
        np.testing.assert_array_equal(first['train_scores'], second['train_scores'])
        self.assertEqual(second['pca'].n_samples_, len(train))

    def test_matches_existing_one_third_pipeline(self):
        train, val = self.data()
        result = analyze_pca(train, val)
        pipeline = make_feature_pipeline(train, reduce=True).fit(train)
        k = result['summary']['retained_components']
        np.testing.assert_allclose(result['validation_scores'].iloc[:, :k], pipeline.transform(val), atol=1e-12)


class KernelPCATests(unittest.TestCase):
    def test_partial_spectrum_uses_total_centered_kernel_variance(self):
        X = np.random.default_rng(42).normal(size=(40, 5))
        model = fit_kernel_pca_rbf(X, n_components=4, gamma=.2)
        variance, ratio = compute_kernel_explained_variance(model, len(X))
        kernel = rbf_kernel(X, gamma=.2)
        H = np.eye(len(X)) - np.ones((len(X), len(X))) / len(X)
        spectrum = np.linalg.eigvalsh(H @ kernel @ H)[::-1]
        np.testing.assert_allclose(variance, spectrum[:4] / (len(X)-1), atol=1e-12)
        np.testing.assert_allclose(ratio, spectrum[:4] / spectrum.sum(), atol=1e-12)
        self.assertLess(ratio.sum(), 1)
        np.testing.assert_allclose(model.transform(X).var(axis=0, ddof=1), variance, atol=1e-12)

    def test_validation_transform_preserves_training_kernel(self):
        X = np.random.default_rng(7).normal(size=(40, 5))
        model = fit_kernel_pca_rbf(X, n_components=4, gamma=.2)
        fitted = model.X_fit_.copy()
        eigenvalues = model.eigenvalues_.copy()
        first = model.transform(X[:3])
        model.transform(np.full((6, 5), 1e8))
        np.testing.assert_array_equal(model.X_fit_, fitted)
        np.testing.assert_array_equal(model.eigenvalues_, eigenvalues)
        np.testing.assert_allclose(model.transform(X[:3]), first)
        with self.assertRaises(ValueError):
            compute_kernel_explained_variance(model, len(X)+1)


if __name__ == '__main__':
    unittest.main()
