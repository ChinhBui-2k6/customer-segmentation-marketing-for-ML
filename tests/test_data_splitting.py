"""Check test isolation, unchanged data and the persisted holdout contract."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import data_splitting as splitting, preprocessing


class DataSplittingTests(unittest.TestCase):
    def raw_data(self):
        records = []
        for group in range(100):
            for member in range(2):
                row = dict.fromkeys(preprocessing.RAW_COLUMNS, 0)
                row.update(ID=group * 2 + member, Year_Birth=1980,
                           Education="Graduation", Marital_Status="Single",
                           Income=float("nan") if group % 3 == 0 else 50000,
                           Dt_Customer="01-01-2014", Recency=group,
                           Z_CostContact=3, Z_Revenue=11,
                           Response=int(group % 5 == 0))
                records.append(row)
        return pd.DataFrame(records)

    def test_test_and_all_experiments_are_disjoint_by_row_and_group(self):
        frame, metadata = preprocessing.create_handoff(self.raw_data())
        plan = splitting.build_split_plan(frame, metadata)
        test = set(plan["test"])
        dev = set(plan["development"])
        self.assertFalse(test & dev)
        self.assertEqual(test | dev, set(range(len(frame))))
        test_groups = set(metadata.iloc[plan["test"]].duplicate_group)
        for name, split in plan["experiments"].items():
            train, validation = set(split["train"]), set(split["validation"])
            self.assertEqual(train | validation, dev)
            self.assertFalse(train & validation or train & test or validation & test)
            train_groups = set(metadata.iloc[split["train"]].duplicate_group)
            val_groups = set(metadata.iloc[split["validation"]].duplicate_group)
            self.assertFalse(train_groups & val_groups or (train_groups | val_groups) & test_groups)
            for rows in split.values():
                self.assertEqual(set(frame.iloc[rows].Response), {0, 1})
            self.assertAlmostEqual(len(validation) / len(dev), int(name.split('_')[1]) / 100, delta=.05)

    def test_test_membership_does_not_depend_on_inner_experiments(self):
        frame, metadata = preprocessing.create_handoff(self.raw_data())
        first = splitting.build_split_plan(frame, metadata)
        second = splitting.build_split_plan(frame, metadata, validation_sizes=(.3,))
        np.testing.assert_array_equal(first["test"], second["test"])
        np.testing.assert_array_equal(first["development"], second["development"])

    def test_export_preserves_source_values_categories_missing_and_dimensions(self):
        raw = self.raw_data()
        original = raw.copy(deep=True)
        frame, metadata = preprocessing.create_handoff(raw)
        plan = splitting.build_split_plan(frame, metadata)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "splits"
            manifest = splitting.export_splits(raw, folder)
            expected = frame.iloc[plan["test"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(pd.read_csv(folder / 'test.csv'), expected)
            raw_rows = metadata.iloc[plan['test']].source_row.to_numpy()
            pd.testing.assert_frame_equal(pd.read_csv(folder / 'test_raw.csv'), raw.iloc[raw_rows].reset_index(drop=True))
            self.assertTrue(expected.Income.isna().any())
            for key in ['imputed', 'encoded', 'scaled', 'dimensionality_reduced']:
                self.assertFalse(manifest[key])
        pd.testing.assert_frame_equal(raw, original)

    def test_rerun_keeps_all_files_and_timestamps_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'splits'
            first = splitting.export_splits(self.raw_data(), folder)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
            second = splitting.export_splits(self.raw_data(), folder)
            after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(before, after)

    def test_changed_data_or_seed_cannot_replace_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'splits'
            raw = self.raw_data()
            splitting.export_splits(raw, folder)
            original = (folder / 'test.csv').read_bytes()
            changed = raw.copy()
            changed.loc[0, 'Income'] = 10000
            with self.assertRaises(ValueError):
                splitting.export_splits(changed, folder)
            with self.assertRaises(ValueError):
                splitting.export_splits(raw, folder, seed=43)
            self.assertEqual(original, (folder / 'test.csv').read_bytes())

    def test_tampering_is_detected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'splits'
            splitting.export_splits(self.raw_data(), folder)
            test = folder / 'test.csv'
            test.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                splitting.export_splits(self.raw_data(), folder)
            self.assertEqual(test.read_bytes(), b'changed')

    def test_misaligned_metadata_and_insufficient_groups_fail(self):
        frame, metadata = preprocessing.create_handoff(self.raw_data())
        with self.assertRaises(ValueError):
            splitting.build_split_plan(frame, metadata.iloc[::-1])
        with self.assertRaises(ValueError):
            splitting.build_split_plan(frame.iloc[:4], metadata.iloc[:4])


if __name__ == '__main__':
    unittest.main()
