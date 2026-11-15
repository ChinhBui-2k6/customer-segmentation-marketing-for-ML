"""Independent checks for score metrics and fold-specific supervised fitting."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_limits

from src.classification import (FoldBalancedGaussianNB, evaluate_classifier,
                                grouped_training_cv, make_classifier, tune_classifier)
from src.evaluation import classification_metrics


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(limits=1)
        X, y = make_classification(n_samples=100, n_features=6, n_informative=4,
                                    n_redundant=0, weights=[.7, .3], random_state=42)
        self.X = pd.DataFrame(np.repeat(X, 2, axis=0), columns=list('abcdef'))
        self.y = np.repeat(y, 2)
        self.groups = np.repeat(np.arange(100), 2)
        self.X.loc[0, 'a'] = np.nan

    def tearDown(self):
        self.limits.restore_original_limits()

    def test_metrics_distinguish_trapezoidal_pr_auc_from_ap(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [.1, .8, .4, .9])
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['f1'], .5)
        self.assertEqual(metrics['roc_auc'], .75)
        self.assertAlmostEqual(metrics['pr_auc'], 19/24)
        self.assertAlmostEqual(metrics['average_precision'], 5/6)
        for key in ['tn', 'fp', 'fn', 'tp']:
            self.assertEqual(metrics[key], 1)

    def test_nb_balancing_recomputed_for_each_fit_and_clone(self):
        X = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
        model = FoldBalancedGaussianNB(balanced=True).fit(X, [0, 0, 0, 0, 1, 1])
        np.testing.assert_allclose(model.class_prior_, [.5, .5])
        other = clone(model).fit(X[:4], [0, 0, 0, 1])
        np.testing.assert_allclose(other.class_count_, [2, 2])
        np.testing.assert_allclose(other.class_prior_, [.5, .5])
        np.testing.assert_allclose(FoldBalancedGaussianNB().fit(X, [0, 0, 0, 0, 1, 1]).class_prior_, [2/3, 1/3])

    def test_cv_keeps_groups_and_covers_all_validation_rows(self):
        folds = grouped_training_cv(self.X, self.y, self.groups, n_splits=3)
        seen = []
        for train, validation in folds:
            self.assertFalse(set(self.groups[train]) & set(self.groups[validation]))
            self.assertEqual(set(self.y[train]), {0, 1})
            seen.extend(validation)
        self.assertEqual(sorted(seen), list(range(len(self.y))))

    def test_tuning_preserves_train_statistics_and_fits_pca_in_folds(self):
        search = tune_classifier(self.X, self.y, self.groups, 'gaussian_nb', reduce=True,
                                 n_splits=2, param_grid={'classifier__var_smoothing': [1e-9, 1e-3]})
        best = search.best_estimator_
        original = best.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer'].statistics_.copy()
        holdout = self.X.iloc[:10].copy() + 1e6
        metrics, _, scores = evaluate_classifier(best, holdout, self.y[:10])
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(metrics['rows'], 10)
        np.testing.assert_allclose(original, self.X.median())
        np.testing.assert_array_equal(original, best.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer'].statistics_)
        # A new fold pipeline learns its own median and PCA sample count.
        fold_train = np.arange(50)
        fitted = clone(best).fit(self.X.iloc[fold_train], self.y[fold_train])
        self.assertEqual(fitted.named_steps['pca'].pca_.n_samples_, 50)
        np.testing.assert_allclose(fitted.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer'].statistics_, self.X.iloc[fold_train].median())

    def test_svm_outputs_margin_and_excludes_target(self):
        estimator = make_classifier(self.X, 'svm_rbf', balanced=True).fit(self.X, self.y)
        _, predictions, scores = evaluate_classifier(estimator, self.X, self.y)
        np.testing.assert_array_equal(predictions, (scores >= 0).astype(int))
        self.assertFalse(hasattr(estimator, 'predict_proba'))
        with self.assertRaises(ValueError):
            make_classifier(self.X.assign(Response=self.y), 'svm_rbf')


if __name__ == '__main__':
    unittest.main()
