"""Regression checks for the cleaning/feature handoff, without the Kaggle file."""
import unittest

import pandas as pd

from src import preprocessing as prep


class PreprocessingTests(unittest.TestCase):
    def sample(self):
        row = {column: 0 for column in prep.RAW_COLUMNS}
        row.update(ID=1, Year_Birth=1980, Education="Graduation", Marital_Status="Single",
                   Income=50000, Dt_Customer="01-01-2014", Z_CostContact=3, Z_Revenue=11,
                   NumWebPurchases=2, NumCatalogPurchases=3, NumStorePurchases=4,
                   NumDealsPurchases=5)
        return pd.DataFrame([row])

    def test_discount_is_not_a_fourth_channel(self):
        features = prep.engineer_features(self.sample())
        self.assertEqual(features.Total_Purchases.iloc[0], 9)

    def test_tenure_does_not_depend_on_other_rows(self):
        first = self.sample()
        second = first.assign(ID=2, Dt_Customer="29-06-2014")
        batch = prep.engineer_features(pd.concat([first, second], ignore_index=True))
        self.assertEqual(batch.Customer_Tenure_Days.iloc[0],
                         prep.engineer_features(first).Customer_Tenure_Days.iloc[0])
        self.assertEqual(batch.Customer_Tenure_Days.iloc[1], 1)

    def test_unknown_marital_status_does_not_invent_household_size(self):
        result = prep.engineer_features(self.sample().assign(Marital_Status="YOLO"))
        self.assertEqual(result.Marital_Group.iloc[0], "Unknown")
        self.assertTrue(result.Family_Size.isna().all())

    def test_missing_income_survives_handoff(self):
        result, _ = prep.create_handoff(self.sample().assign(Income=float("nan")))
        self.assertTrue(result.Income.isna().all())
        self.assertTrue(pd.api.types.is_object_dtype(result.Education_Level))

    def test_groups_do_not_use_response_and_metadata_is_not_a_feature(self):
        first = self.sample()
        second = first.assign(ID=2, Response=1)
        raw = pd.concat([first, second], ignore_index=True)
        original = raw.copy(deep=True)
        result, metadata = prep.create_handoff(raw)
        self.assertEqual(metadata.duplicate_group.nunique(), 1)
        self.assertEqual(metadata.source_row.tolist(), [0, 1])
        self.assertEqual(result.Response.tolist(), [0, 1])
        self.assertTrue({"ID", "customer_id", "duplicate_group"}.isdisjoint(result.columns))
        pd.testing.assert_frame_equal(raw, original)

    def test_outlier_boundaries_and_row_mapping(self):
        raw = pd.concat([self.sample().assign(ID=1, Year_Birth=1893),
                         self.sample().assign(ID=2, Income=200000),
                         self.sample().assign(ID=3, Income=200001),
                         self.sample().assign(ID=4, Income=-1)], ignore_index=True)
        result, metadata = prep.create_handoff(raw)
        self.assertEqual(metadata.customer_id.tolist(), [2, 4])
        self.assertEqual(metadata.source_row.tolist(), [1, 3])
        self.assertTrue(pd.isna(result.Income.iloc[1]))

    def test_rejects_missing_schema_invalid_target_and_future_date(self):
        for frame in [self.sample().drop(columns="Response"),
                      self.sample().assign(Response=2),
                      self.sample().assign(Dt_Customer="01-01-2026")]:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaises(ValueError):
                    prep.create_handoff(frame)


if __name__ == "__main__":
    unittest.main()
