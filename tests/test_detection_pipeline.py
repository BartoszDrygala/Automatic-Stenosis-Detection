"""
Tests for detect_stenosis_combined / detect_all_stenosis on synthetic vessels
where the ground truth (where the narrowing is, how severe) is known.
"""
import cv2
import numpy as np
import pytest

from helpers import dip_profile, make_vessel_mask, profile_to_path_dist, run_detection
from stenosis import detect_all_stenosis, detect_stenosis_combined

REQUIRED_KEYS = {
    "path", "idx", "orig_idx", "severity", "contour",
    "left", "right", "g_set", "l_set",
}


#***************************************************************
# detect_stenosis_combined (single path)
#***************************************************************

class TestDetectCombined:
    def test_flat_vessel_gives_no_selection(self):
        path, dist = profile_to_path_dist(np.full(120, 10.0))
        selected, g_set, l_set = detect_stenosis_combined(path, dist)
        assert selected == []
        assert not g_set and not l_set

    def test_short_path_gives_no_selection(self):
        path, dist = profile_to_path_dist(np.full(20, 10.0))
        selected, _, _ = detect_stenosis_combined(path, dist)
        assert selected == []

    def test_dip_is_found_at_the_right_place(self):
        # flat-bottomed dip: the filters reject sharp V shapes (see is_wide_valley)
        profile = np.full(150, 10.0)
        profile[65:85] = 4.0
        path, dist = profile_to_path_dist(profile)
        selected, g_set, l_set = detect_stenosis_combined(path, dist)

        assert len(selected) >= 1
        refined_idx, severity, _ = selected[0]
        assert 65 <= refined_idx < 85
        assert severity == pytest.approx(0.6, abs=0.05)
        assert g_set and l_set

    def test_at_most_two_lesions_are_returned(self):
        profile = np.full(300, 10.0)
        for start in (50, 110, 170, 230):
            profile[start:start + 20] = 4.0
        path, dist = profile_to_path_dist(profile)
        selected, _, _ = detect_stenosis_combined(path, dist)
        assert len(selected) <= 2

    def test_selected_lesions_are_at_least_20_samples_apart(self):
        profile = np.full(300, 10.0)
        for start in (50, 110, 170, 230):
            profile[start:start + 20] = 4.0
        path, dist = profile_to_path_dist(profile)
        selected, _, _ = detect_stenosis_combined(path, dist)
        positions = [s[0] for s in selected]
        for i, a in enumerate(positions):
            for b in positions[i + 1:]:
                assert abs(a - b) >= 20

    def test_severity_is_at_least_the_minimum_threshold(self):
        profile = np.full(150, 10.0)
        profile[65:85] = 4.0
        path, dist = profile_to_path_dist(profile)
        selected, _, _ = detect_stenosis_combined(path, dist)
        assert all(s[1] >= 0.35 for s in selected)

    def test_shallow_dip_is_below_the_severity_threshold(self):
        profile = np.full(150, 10.0)
        profile[65:85] = 8.5  # only 15 % narrower
        path, dist = profile_to_path_dist(profile)
        selected, _, _ = detect_stenosis_combined(path, dist)
        assert selected == []

    def test_gradual_taper_is_not_reported(self):
        """A vessel that simply gets thinner towards the end isn't a stenosis."""
        profile = np.linspace(12.0, 4.0, 150)
        path, dist = profile_to_path_dist(profile)
        selected, _, _ = detect_stenosis_combined(path, dist)
        assert selected == []


#***************************************************************
# detect_all_stenosis (full chain on synthetic masks)
#***************************************************************

class TestDetectAll:
    def test_no_paths_gives_no_detections(self):
        mask = np.zeros((50, 50), np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        assert detect_all_stenosis([], dist, mask) == []

    def test_healthy_vessel_has_no_detections(self, healthy_mask):
        *_, detections = run_detection(healthy_mask)
        assert detections == []

    def test_stenotic_vessel_has_a_detection(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        assert len(detections) >= 1

    def test_detection_has_all_expected_keys(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        for det in detections:
            assert REQUIRED_KEYS <= set(det)

    def test_detection_is_located_at_the_narrowing(self, stenotic_mask):
        _, paths, _, detections = run_detection(stenotic_mask)
        det = detections[0]
        x, y = det["path"][det["idx"]]
        # narrowing is centred at x = 20 + 150 = 170
        assert abs(x - 170) <= 10
        assert y == 50

    def test_severity_matches_the_geometry(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        # healthy radius ~11, narrowest ~4 -> ~0.64
        assert detections[0]["severity"] == pytest.approx(0.64, abs=0.1)

    def test_severity_is_a_fraction_between_threshold_and_one(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        assert all(0.35 <= d["severity"] <= 1.0 for d in detections)

    def test_region_brackets_the_detection_index(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        for det in detections:
            assert det["left"] <= det["idx"] <= det["right"]

    def test_contour_is_a_valid_opencv_contour_inside_the_vessel(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        contour = detections[0]["contour"]
        assert contour.ndim == 3 and contour.shape[1:] == (1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        assert (stenotic_mask[y:y + h, x:x + w] > 0).any()
        assert cv2.contourArea(contour) > 0

    def test_contour_surrounds_the_narrowing(self, stenotic_mask):
        *_, detections = run_detection(stenotic_mask)
        contour = detections[0]["contour"]
        assert cv2.pointPolygonTest(contour, (170.0, 50.0), False) >= 0

    def test_deeper_narrowing_gives_higher_severity(self):
        mild = make_vessel_mask(centers=(150,), narrow_half=4)
        severe = make_vessel_mask(centers=(150,), narrow_half=2, sigma=8)
        *_, d_mild = run_detection(mild)
        *_, d_severe = run_detection(severe)
        assert d_mild and d_severe
        assert d_severe[0]["severity"] > d_mild[0]["severity"]

    def test_two_distant_narrowings_are_reported_separately(self):
        mask = make_vessel_mask(centers=(100, 300), length=400)
        *_, detections = run_detection(mask)
        assert len(detections) == 2
        xs = sorted(d["path"][d["idx"]][0] for d in detections)
        assert abs(xs[0] - 120) <= 15
        assert abs(xs[1] - 320) <= 15

    def test_detections_are_not_mutating_the_input_mask(self, stenotic_mask):
        before = stenotic_mask.copy()
        run_detection(stenotic_mask)
        assert np.array_equal(stenotic_mask, before)

    def test_result_is_deterministic(self, stenotic_mask):
        *_, first = run_detection(stenotic_mask)
        *_, second = run_detection(stenotic_mask)
        assert [(d["idx"], d["severity"]) for d in first] == \
               [(d["idx"], d["severity"]) for d in second]

    def test_vessel_without_enough_length_is_not_analysed(self):
        mask = make_vessel_mask(centers=(15,), length=40)
        *_, detections = run_detection(mask)
        assert detections == []


class TestFinalMerging:
    """Nearby lesions are merged into one so the user sees a single region."""

    def test_two_close_narrowings_become_one_detection(self):
        mask = make_vessel_mask(centers=(150, 215), length=400)
        *_, detections = run_detection(mask)
        assert len(detections) == 1

    def test_merged_detection_keeps_the_highest_severity(self):
        mask = make_vessel_mask(centers=(150, 215), length=400)
        *_, merged = run_detection(mask)
        single = make_vessel_mask(centers=(150,), length=400)
        *_, one = run_detection(single)
        assert merged[0]["severity"] >= one[0]["severity"] - 1e-6

    def test_merged_region_spans_both_narrowings(self):
        mask = make_vessel_mask(centers=(150, 215), length=400)
        *_, detections = run_detection(mask)
        det = detections[0]
        assert det["right"] - det["left"] > 60

    def test_results_are_ordered_by_decreasing_severity_before_merging(self):
        mask = make_vessel_mask(centers=(100, 300), length=400)
        *_, detections = run_detection(mask)
        severities = [d["severity"] for d in detections]
        assert severities == sorted(severities, reverse=True)


#***************************************************************
# Real data shipped with the repo
#***************************************************************

class TestRealMasks:
    """Smoke / regression tests on the example data in data/masks."""

    @pytest.mark.parametrize("name", ["1", "2", "3", "11", "12", "13", "14", "15"])
    def test_runs_and_returns_well_formed_detections(self, name, data_dir):
        mask = cv2.imread(str(data_dir / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        skeleton, paths, dist, detections = run_detection(mask)

        assert skeleton.shape == mask.shape
        assert len(paths) > 0
        for det in detections:
            assert REQUIRED_KEYS <= set(det)
            assert 0.0 <= det["severity"] <= 1.0
            assert 0 <= det["left"] <= det["right"] < len(det["path"])
            x, y = det["path"][det["idx"]]
            assert mask[y, x] > 0, "detection must lie on the vessel"

    def test_detections_have_distinct_contours_after_merging(self, data_dir):
        """After final merging, no two detections should still be close."""
        from stenosis import contours_close_or_overlap

        mask = cv2.imread(str(data_dir / "masks" / "15.png"), cv2.IMREAD_GRAYSCALE)
        *_, detections = run_detection(mask)
        for i, a in enumerate(detections):
            for b in detections[i + 1:]:
                assert not contours_close_or_overlap(
                    a["contour"], b["contour"], distance_threshold=20
                )
