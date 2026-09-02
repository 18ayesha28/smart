"""
Unit tests for treatment_response.py.

Run with:
    python -m unittest test_treatment_response.py -v
"""

import unittest

from treatment_response import (
    VisitMeasurement,
    classify_response,
    CR, PR, SD, PD, INSUFFICIENT_DATA,
)


def visit(date, product=None, area=None, tumor_type="glioma", max_d=None, perp_d=None):
    return VisitMeasurement(
        scan_date=date, tumor_type=tumor_type,
        max_diameter_mm=max_d, perpendicular_diameter_mm=perp_d,
        product_bidirectional_mm2=product, area_mm2=area,
    )


class TestNoOrSingleVisit(unittest.TestCase):
    def test_empty_history(self):
        result = classify_response([])
        self.assertEqual(result.category, INSUFFICIENT_DATA)
        self.assertIsNone(result.current)

    def test_single_visit_is_baseline_only(self):
        result = classify_response([visit("2026-01-01", product=300.0)])
        self.assertEqual(result.category, INSUFFICIENT_DATA)
        self.assertIn("Baseline established", result.rationale)
        self.assertIsNone(result.baseline)
        self.assertIsNone(result.pct_change_from_baseline)


class TestTwoVisitCategories(unittest.TestCase):
    def test_clear_partial_response(self):
        # 300 -> 120 is -60%, past the -50% PR threshold, and well short of +25% from nadir.
        history = [visit("2026-01-01", product=300.0), visit("2026-02-01", product=120.0)]
        result = classify_response(history)
        self.assertEqual(result.category, PR)
        self.assertAlmostEqual(result.pct_change_from_baseline, -60.0, places=3)
        self.assertAlmostEqual(result.pct_change_from_nadir, -60.0, places=3)

    def test_clear_progressive_disease(self):
        # 200 -> 260 is +30%, past the +25% PD-vs-nadir threshold.
        history = [visit("2026-01-01", product=200.0), visit("2026-02-01", product=260.0)]
        result = classify_response(history)
        self.assertEqual(result.category, PD)
        self.assertAlmostEqual(result.pct_change_from_nadir, 30.0, places=3)

    def test_stable_disease_small_change(self):
        # 200 -> 210 is +5%: neither PD (>=25%) nor PR (<=-50%).
        history = [visit("2026-01-01", product=200.0), visit("2026-02-01", product=210.0)]
        result = classify_response(history)
        self.assertEqual(result.category, SD)

    def test_boundary_exactly_25pct_is_pd(self):
        history = [visit("2026-01-01", product=200.0), visit("2026-02-01", product=250.0)]  # +25.0% exactly
        result = classify_response(history)
        self.assertEqual(result.category, PD)

    def test_boundary_just_under_25pct_is_sd(self):
        history = [visit("2026-01-01", product=200.0), visit("2026-02-01", product=249.0)]  # +24.5%
        result = classify_response(history)
        self.assertEqual(result.category, SD)

    def test_boundary_exactly_50pct_decrease_is_pr(self):
        history = [visit("2026-01-01", product=200.0), visit("2026-02-01", product=100.0)]  # -50.0% exactly
        result = classify_response(history)
        self.assertEqual(result.category, PR)


class TestBaselineVsPreviousVsNadirAreDistinct(unittest.TestCase):
    """The critical case the spec calls out: previous != nadir, and the
    category decision must be driven by nadir (progression) / baseline
    (response), never by conflating nadir with 'previous visit'."""

    def test_four_visit_trajectory(self):
        # baseline=300, v2=150 (becomes nadir), v3=220 (previous), v4=180 (current)
        history = [
            visit("2026-01-01", product=300.0),
            visit("2026-03-01", product=150.0),
            visit("2026-05-01", product=220.0),
            visit("2026-07-01", product=180.0),
        ]
        result = classify_response(history)

        self.assertEqual(result.baseline.scan_date, "2026-01-01")
        self.assertEqual(result.previous.scan_date, "2026-05-01")
        self.assertEqual(result.nadir.scan_date, "2026-03-01")
        self.assertNotEqual(result.previous.scan_date, result.nadir.scan_date)

        # vs previous (220): (180-220)/220 = -18.18% (looks like improvement)
        self.assertAlmostEqual(result.pct_change_from_previous, -18.1818, places=3)
        # vs nadir (150): (180-150)/150 = +20% (not yet PD, but rising off the low point)
        self.assertAlmostEqual(result.pct_change_from_nadir, 20.0, places=3)
        # vs baseline (300): (180-300)/300 = -40% (not yet PR's -50%)
        self.assertAlmostEqual(result.pct_change_from_baseline, -40.0, places=3)

        # None of the three references would give the same verdict if conflated:
        # nadir-based would almost call PD (20% < 25%, so still SD), baseline-based
        # is not PR (-40% > -50%) -> overall correctly SD.
        self.assertEqual(result.category, SD)

    def test_nadir_persists_across_multiple_prior_visits(self):
        history = [
            visit("2026-01-01", product=400.0),
            visit("2026-02-01", product=100.0),   # lowest point (nadir)
            visit("2026-03-01", product=300.0),
            visit("2026-04-01", product=126.0),   # current: +26% vs nadir(100) -> PD
        ]
        result = classify_response(history)
        self.assertEqual(result.nadir.scan_date, "2026-02-01")
        self.assertEqual(result.category, PD)
        self.assertAlmostEqual(result.pct_change_from_nadir, 26.0, places=3)


class TestZeroAndMissingMeasurements(unittest.TestCase):
    def test_new_lesion_after_no_prior_measurable_disease(self):
        history = [visit("2026-01-01", product=None, area=None), visit("2026-02-01", product=180.0)]
        result = classify_response(history)
        self.assertEqual(result.category, PD)
        self.assertIn("new measurable lesion", result.rationale.lower())

    def test_no_measurable_disease_at_any_point_is_stable(self):
        history = [visit("2026-01-01", product=None, area=None), visit("2026-02-01", product=None, area=None)]
        result = classify_response(history)
        self.assertEqual(result.category, SD)

    def test_complete_response_when_current_has_no_measurable_tumor(self):
        history = [visit("2026-01-01", product=300.0), visit("2026-02-01", product=None, area=None)]
        result = classify_response(history)
        self.assertEqual(result.category, CR)
        self.assertIn("does not rule out", result.rationale.lower())

    def test_area_fallback_when_product_missing(self):
        history = [
            visit("2026-01-01", product=None, area=300.0),
            visit("2026-02-01", product=None, area=120.0),
        ]
        result = classify_response(history)
        self.assertEqual(result.metric_used, "area_mm2")
        self.assertEqual(result.category, PR)

    def test_no_division_by_zero_crash_with_zero_area(self):
        # area_mm2 == 0.0 is treated as "no measurable tumor" (not a divide-by-zero crash).
        history = [visit("2026-01-01", product=None, area=0.0), visit("2026-02-01", product=150.0)]
        result = classify_response(history)  # must not raise
        self.assertEqual(result.category, PD)


class TestDomainMismatchCaveat(unittest.TestCase):
    def test_glioma_only_history_has_no_domain_caveat(self):
        history = [
            visit("2026-01-01", product=300.0, tumor_type="glioma"),
            visit("2026-02-01", product=150.0, tumor_type="glioma"),
        ]
        result = classify_response(history)
        self.assertEqual(result.caveats, [])

    def test_meningioma_triggers_segmentation_domain_caveat(self):
        history = [
            visit("2026-01-01", product=300.0, tumor_type="meningioma"),
            visit("2026-02-01", product=150.0, tumor_type="meningioma"),
        ]
        result = classify_response(history)
        self.assertTrue(any("LGG" in c for c in result.caveats))

    def test_tumor_type_change_across_visits_triggers_caveat(self):
        history = [
            visit("2026-01-01", product=300.0, tumor_type="glioma"),
            visit("2026-02-01", product=150.0, tumor_type="meningioma"),
        ]
        result = classify_response(history)
        self.assertTrue(any("differs across visits" in c for c in result.caveats))


class TestInvalidPatientIdHandledByCaller(unittest.TestCase):
    """classify_response itself is patient-agnostic (it only sees a history
    list) — invalid/blank patient_id validation lives in patient_history.py
    and is covered in test_patient_history.py."""

    def test_classify_response_does_not_care_about_patient_identity(self):
        history = [visit("2026-01-01", product=300.0), visit("2026-02-01", product=150.0)]
        result = classify_response(history)
        self.assertIn(result.category, {CR, PR, SD, PD})


if __name__ == "__main__":
    unittest.main()
