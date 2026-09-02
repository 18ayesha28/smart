"""
Unit tests for patient_history.py.

Uses a temporary SQLite file per test (not ':memory:') specifically to also
verify that data survives being reopened — i.e. that persistence actually
works across what would be separate Streamlit process runs.

Run with:
    python -m unittest test_patient_history.py -v
"""

import os
import tempfile
import unittest

from patient_history import (
    record_visit, get_visit_history, get_all_patient_ids, delete_patient_history,
)


class PatientHistoryTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(self.db_path)  # let record_visit create it fresh

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


class TestFirstSecondThirdVisit(PatientHistoryTestCase):
    def test_first_visit(self):
        vid = record_visit("P-001", "2026-01-01", tumor_type="glioma",
                            product_bidirectional_mm2=300.0, db_path=self.db_path)
        self.assertIsInstance(vid, int)
        history = get_visit_history("P-001", db_path=self.db_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].scan_date, "2026-01-01")

    def test_second_visit_appends_not_overwrites(self):
        record_visit("P-001", "2026-01-01", product_bidirectional_mm2=300.0, db_path=self.db_path)
        record_visit("P-001", "2026-02-01", product_bidirectional_mm2=150.0, db_path=self.db_path)
        history = get_visit_history("P-001", db_path=self.db_path)
        self.assertEqual(len(history), 2)
        self.assertEqual([v.scan_date for v in history], ["2026-01-01", "2026-02-01"])

    def test_third_visit_and_date_ordering_independent_of_insertion_order(self):
        # Insert out of chronological order — retrieval must still be date-sorted.
        record_visit("P-001", "2026-03-01", product_bidirectional_mm2=200.0, db_path=self.db_path)
        record_visit("P-001", "2026-01-01", product_bidirectional_mm2=300.0, db_path=self.db_path)
        record_visit("P-001", "2026-02-01", product_bidirectional_mm2=150.0, db_path=self.db_path)
        history = get_visit_history("P-001", db_path=self.db_path)
        self.assertEqual([v.scan_date for v in history], ["2026-01-01", "2026-02-01", "2026-03-01"])


class TestMultiplePatients(PatientHistoryTestCase):
    def test_visits_are_isolated_per_patient(self):
        record_visit("P-001", "2026-01-01", product_bidirectional_mm2=300.0, db_path=self.db_path)
        record_visit("P-002", "2026-01-01", product_bidirectional_mm2=999.0, db_path=self.db_path)
        record_visit("P-001", "2026-02-01", product_bidirectional_mm2=150.0, db_path=self.db_path)

        history_1 = get_visit_history("P-001", db_path=self.db_path)
        history_2 = get_visit_history("P-002", db_path=self.db_path)
        self.assertEqual(len(history_1), 2)
        self.assertEqual(len(history_2), 1)

    def test_new_patient_has_empty_history(self):
        record_visit("P-001", "2026-01-01", product_bidirectional_mm2=300.0, db_path=self.db_path)
        self.assertEqual(get_visit_history("P-999-NEW", db_path=self.db_path), [])

    def test_get_all_patient_ids(self):
        record_visit("P-001", "2026-01-01", db_path=self.db_path)
        record_visit("P-002", "2026-01-01", db_path=self.db_path)
        record_visit("P-001", "2026-02-01", db_path=self.db_path)
        self.assertEqual(get_all_patient_ids(db_path=self.db_path), ["P-001", "P-002"])


class TestInvalidInput(PatientHistoryTestCase):
    def test_blank_patient_id_raises(self):
        with self.assertRaises(ValueError):
            record_visit("", "2026-01-01", db_path=self.db_path)

    def test_whitespace_only_patient_id_raises(self):
        with self.assertRaises(ValueError):
            record_visit("   ", "2026-01-01", db_path=self.db_path)

    def test_blank_scan_date_raises(self):
        with self.assertRaises(ValueError):
            record_visit("P-001", "", db_path=self.db_path)

    def test_get_history_for_blank_patient_id_returns_empty(self):
        self.assertEqual(get_visit_history("", db_path=self.db_path), [])

    def test_missing_measurements_are_stored_as_none(self):
        record_visit("P-001", "2026-01-01", tumor_type=None, db_path=self.db_path)
        history = get_visit_history("P-001", db_path=self.db_path)
        self.assertIsNone(history[0].product_bidirectional_mm2)
        self.assertIsNone(history[0].tumor_type)


class TestDeletion(PatientHistoryTestCase):
    def test_delete_patient_history(self):
        record_visit("P-001", "2026-01-01", db_path=self.db_path)
        record_visit("P-001", "2026-02-01", db_path=self.db_path)
        removed = delete_patient_history("P-001", db_path=self.db_path)
        self.assertEqual(removed, 2)
        self.assertEqual(get_visit_history("P-001", db_path=self.db_path), [])


class TestPersistenceAcrossReconnect(PatientHistoryTestCase):
    """Simulates 'restart Streamlit' by dropping all Python-level references
    and reconnecting to the same on-disk file fresh."""

    def test_data_survives_reconnect(self):
        record_visit("P-RESTART", "2026-01-01", product_bidirectional_mm2=300.0, db_path=self.db_path)
        record_visit("P-RESTART", "2026-02-01", product_bidirectional_mm2=150.0, db_path=self.db_path)

        # New, independent connection to the same file path — nothing shared in-process.
        history = get_visit_history("P-RESTART", db_path=self.db_path)
        self.assertEqual(len(history), 2)
        self.assertAlmostEqual(history[1].product_bidirectional_mm2, 150.0)


if __name__ == "__main__":
    unittest.main()
