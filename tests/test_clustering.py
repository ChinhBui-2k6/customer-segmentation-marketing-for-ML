"""K-Means selection/fit isolation and positional profiling checks."""
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from threadpoolctl import threadpool_limits

from src.clustering import analyze_kmeans, cluster_metrics, describe_clusters


class KMeansTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        data, _ = make_blobs(n_samples=120, centers=[[-8, -8], [8, 8]],
                             cluster_std=.25, random_state=12)
        self.train = pd.DataFrame(data[:90], columns=["Income", "Age"])
        self.val = pd.DataFrame(data[90:], columns=self.train.columns)

    def tearDown(self):
        self.threads.restore_original_limits()

    def test_selects_two_well_separated_blobs(self):
        result = analyze_kmeans(self.train, self.val, k_values=(2, 3, 4))
        self.assertEqual(result["selected_k"], 2)
        self.assertGreater(result["candidates"].iloc[0].silhouette, .9)
        np.testing.assert_array_equal(result["pipeline"].predict(self.val), result["validation_labels"])
        self.assertEqual(result["evaluation"].partition.tolist(), ["train", "validation"])

    def test_extreme_validation_does_not_change_training_or_selection(self):
        a = analyze_kmeans(self.train, self.val, k_values=(2, 3))
        b = analyze_kmeans(self.train, self.val + 1e6, k_values=(2, 3))
        pd.testing.assert_frame_equal(a["candidates"], b["candidates"])
        np.testing.assert_array_equal(a["model"].cluster_centers_, b["model"].cluster_centers_)
        self.assertEqual(a["selected_k"], b["selected_k"])

    def test_profile_positional_alignment_and_label_not_in_features(self):
        features = pd.DataFrame({"Income": [10, 20, 30, 40], "Category": ["A", "A", "B", "B"]}, index=[4, 8, 12, 16])
        profile = describe_clusters(features, pd.Series([0, 1, 1, 1], index=[9, 7, 5, 3]), [0, 0, 1, 1])
        self.assertEqual(profile["summary"].loc[1, "response_rate"], 1)
        self.assertEqual(profile["numeric_median"].loc[0, "Income"], 15)
        self.assertEqual(profile["categories"].fraction.sum(), 2)
        with self.assertRaises(ValueError):
            analyze_kmeans(self.train.assign(Response=0), self.val.assign(Response=0))

    def test_invalid_k_and_undefined_metrics(self):
        for values in [(), (1, 2), (2, 2), (2.5,), (len(self.train),)]:
            with self.assertRaises(ValueError):
                analyze_kmeans(self.train, self.val, k_values=values)
        self.assertTrue(np.isnan(cluster_metrics(np.zeros((5, 2)), np.zeros(5))["silhouette"]))


if __name__ == "__main__":
    unittest.main()
