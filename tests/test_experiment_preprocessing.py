"""Adversarial holdout checks for learned preprocessing state."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.experiment_preprocessing import make_feature_pipeline


class ExperimentPreprocessingTests(unittest.TestCase):
    def train(self):
        return pd.DataFrame({"Income": [10., 20., np.nan, 30., 40., 50.],
                             "Empty": [np.nan] * 6,
                             "Age": [20., 25., 30., 35., 40., 45.],
                             "Category": ["A", "B", "A", "B", "A", "B"]})

    def test_holdout_cannot_change_imputer_scaler_categories_or_pca(self):
        train = self.train()
        pipe = make_feature_pipeline(train, reduce=True)
        transformed = pipe.fit_transform(train)
        pre = pipe.named_steps["preprocessor"]
        num = pre.named_transformers_["num"]
        self.assertEqual(num.named_steps["imputer"].statistics_[0], 30.)
        np.testing.assert_allclose(num.named_steps["scaler"].mean_, [30., 0., 32.5])
        before = pipe.named_steps["pca"].pca_.components_.copy()
        holdout = train.iloc[:2].copy()
        holdout["Income"] = [1e9, np.nan]
        holdout["Category"] = "UNSEEN"
        self.assertTrue(np.isfinite(pipe.transform(holdout)).all())
        self.assertEqual(transformed.shape, (6, 1))
        self.assertEqual(pipe.named_steps["pca"].pca_.n_samples_, 6)
        np.testing.assert_array_equal(before, pipe.named_steps["pca"].pca_.components_)
        self.assertNotIn("UNSEEN", pre.named_transformers_["cat"].named_steps["onehot"].categories_[0])
        self.assertEqual(num.named_steps["imputer"].statistics_[0], 30.)

    def test_cloned_pipeline_refits_each_fold(self):
        train = self.train()
        original = make_feature_pipeline(train, reduce=True).fit(train)
        shifted = train.copy()
        shifted["Income"] += 100
        other = clone(original).fit(shifted)
        self.assertEqual(other.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].statistics_[0], 130.)
        self.assertEqual(original.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].statistics_[0], 30.)

    def test_reject_target_or_metadata(self):
        for name in ["Response", "duplicate_group", "ID", "processed_row"]:
            with self.assertRaises(ValueError):
                make_feature_pipeline(self.train().assign(**{name: 0}))


if __name__ == "__main__":
    unittest.main()
