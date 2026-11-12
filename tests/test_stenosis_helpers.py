"""
Unit tests for the small building blocks in stenosis.py.

These work on 1D radius profiles (what the distance transform gives along a
vessel centre line), so every expected value can be checked by hand.
"""
import cv2
import numpy as np
import pytest

from helpers import dip_profile, profile_to_path_dist, rect_contour
from stenosis import (
    build_exact_stenosis_mask,
    compute_stenosis_severity,
    contours_close_or_overlap,
    contrast_filter,
    detect_stenosis_global,
    detect_stenosis_local_simple,
    extract_full_stenosis_region,
    get_main_contour,
    is_wide_valley,
    merge_contours,
    not_tapering,
    refine_to_minimum,
    split_path,
)

FLAT = np.full(100, 10.0)
DIP = dip_profile()  # base 10, dips to 3 at index 50


#***************************************************************
# detect_stenosis_global
#***************************************************************

class TestDetectGlobal:
    def test_short_path_returns_nothing(self):
        path, dist = profile_to_path_dist(np.full(29, 10.0))
        idx, values, score = detect_stenosis_global(path, dist)
        assert len(idx) == 0
        assert len(values) == 29
        assert score is None

    def test_all_zero_radius_returns_nothing(self):
        path, dist = profile_to_path_dist(np.zeros(60))
        idx, _, score = detect_stenosis_global(path, dist)
        assert len(idx) == 0
        assert score is None

    def test_flat_profile_has_no_detections(self):
        path, dist = profile_to_path_dist(FLAT)
        idx, _, _ = detect_stenosis_global(path, dist)
        assert len(idx) == 0

    def test_dip_is_detected_around_its_centre(self):
        path, dist = profile_to_path_dist(DIP)
        idx, smooth, drop = detect_stenosis_global(path, dist)
        assert len(idx) > 0
        assert min(idx) <= 50 <= max(idx)
        assert len(drop) == len(smooth) == 100
        assert drop[50] > 0.35

    def test_edges_of_the_path_are_never_reported(self):
        path, dist = profile_to_path_dist(DIP)
        idx, _, _ = detect_stenosis_global(path, dist)
        assert all(20 < i < 80 for i in idx)

    def test_dip_near_the_start_is_ignored(self):
        path, dist = profile_to_path_dist(dip_profile(center=8))
        idx, _, _ = detect_stenosis_global(path, dist)
        assert not any(i <= 20 for i in idx)

    def test_higher_threshold_gives_fewer_detections(self):
        path, dist = profile_to_path_dist(DIP)
        low, _, _ = detect_stenosis_global(path, dist, drop_thresh=0.2)
        high, _, _ = detect_stenosis_global(path, dist, drop_thresh=0.5)
        assert len(high) < len(low)

    def test_impossible_threshold_gives_nothing(self):
        path, dist = profile_to_path_dist(DIP)
        idx, _, _ = detect_stenosis_global(path, dist, drop_thresh=0.99)
        assert len(idx) == 0


#***************************************************************
# detect_stenosis_local_simple
#***************************************************************

class TestDetectLocal:
    def test_short_path_returns_nothing(self):
        path, dist = profile_to_path_dist(np.full(29, 10.0))
        idx, values, third = detect_stenosis_local_simple(path, dist)
        assert len(idx) == 0 and len(values) == 29 and third is None

    def test_flat_profile_has_no_detections(self):
        path, dist = profile_to_path_dist(FLAT)
        idx, _, _ = detect_stenosis_local_simple(path, dist)
        assert len(idx) == 0

    def test_dip_is_detected_near_its_centre(self):
        path, dist = profile_to_path_dist(DIP)
        idx, _, _ = detect_stenosis_local_simple(path, dist)
        assert len(idx) > 0
        assert any(abs(i - 50) <= 5 for i in idx)

    def test_only_indices_with_a_full_window_are_checked(self):
        path, dist = profile_to_path_dist(DIP)
        idx, _, _ = detect_stenosis_local_simple(path, dist, window=15)
        assert all(15 <= i < 100 - 15 for i in idx)

    def test_smoothed_values_are_returned(self):
        path, dist = profile_to_path_dist(DIP)
        _, values, _ = detect_stenosis_local_simple(path, dist)
        assert len(values) == 100


#***************************************************************
# contrast_filter
#***************************************************************

class TestContrastFilter:
    def test_dip_passes(self):
        assert contrast_filter(50, DIP)

    def test_flat_profile_fails(self):
        assert not contrast_filter(50, FLAT)

    def test_index_too_close_to_the_start_fails(self):
        assert not contrast_filter(1, DIP)

    def test_index_too_close_to_the_end_fails(self):
        assert not contrast_filter(len(DIP) - 1, DIP)

    def test_min_drop_is_respected(self):
        assert not contrast_filter(50, DIP, min_drop=0.9)


#***************************************************************
# not_tapering
#***************************************************************

class TestNotTapering:
    def test_flat_profile_is_not_tapering(self):
        assert not_tapering(50, FLAT)

    def test_vessel_getting_narrower_is_tapering(self):
        values = np.array([10.0] * 30 + [5.0] * 30)
        assert not not_tapering(30, values)

    def test_vessel_getting_wider_is_not_tapering(self):
        values = np.array([5.0] * 30 + [10.0] * 30)
        assert not_tapering(30, values)

    def test_too_close_to_the_edge_is_rejected(self):
        assert not not_tapering(2, FLAT)
        assert not not_tapering(len(FLAT) - 2, FLAT)


#***************************************************************
# refine_to_minimum
#***************************************************************

class TestRefineToMinimum:
    def test_moves_to_the_nearby_minimum(self):
        assert refine_to_minimum(45, DIP) == 50

    def test_stays_when_already_at_the_minimum(self):
        assert refine_to_minimum(50, DIP) == 50

    def test_works_close_to_the_start(self):
        values = np.array([5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert refine_to_minimum(0, values) == 2

    def test_does_not_look_further_than_the_window(self):
        values = np.full(60, 10.0)
        values[50] = 1.0  # minimum is 40 samples away
        assert refine_to_minimum(10, values, window=10) != 50


#***************************************************************
# compute_stenosis_severity
#***************************************************************

class TestSeverity:
    def test_value_for_a_known_dip(self):
        # healthy radius 10, narrowest point 3 -> (10 - 3) / 10
        assert compute_stenosis_severity(50, DIP) == pytest.approx(0.7)

    def test_flat_profile_has_zero_severity(self):
        assert compute_stenosis_severity(50, FLAT) == 0.0

    def test_edges_give_zero(self):
        assert compute_stenosis_severity(2, DIP) == 0.0
        assert compute_stenosis_severity(len(DIP) - 2, DIP) == 0.0

    def test_all_zero_profile_gives_zero_instead_of_dividing_by_zero(self):
        assert compute_stenosis_severity(50, np.zeros(100)) == 0.0

    def test_deeper_dip_means_higher_severity(self):
        shallow = compute_stenosis_severity(50, dip_profile(depth=3))
        deep = compute_stenosis_severity(50, dip_profile(depth=8))
        assert deep > shallow

    def test_result_is_a_fraction(self):
        assert 0.0 <= compute_stenosis_severity(50, DIP) <= 1.0


#***************************************************************
# is_wide_valley
#***************************************************************

class TestIsWideValley:
    def test_flat_profile_counts_as_wide(self):
        assert is_wide_valley(50, FLAT)

    def test_flat_bottomed_valley_is_wide(self):
        values = np.full(100, 10.0)
        values[40:60] = 3.0
        assert is_wide_valley(50, values)

    def test_sharp_v_shaped_dip_is_not_wide(self):
        assert not is_wide_valley(50, DIP)


#***************************************************************
# split_path
#***************************************************************

class TestSplitPath:
    @staticmethod
    def _path(n):
        return [(i, 0) for i in range(n)]

    def test_offsets_follow_segment_length_minus_overlap(self):
        segments = split_path(self._path(200))
        assert [offset for _, offset in segments] == [0, 60, 120]

    def test_segment_lengths(self):
        segments = split_path(self._path(200))
        assert [len(seg) for seg, _ in segments] == [90, 90, 80]

    def test_consecutive_segments_overlap_by_the_requested_amount(self):
        (a, _), (b, _) = split_path(self._path(200), segment_length=90, overlap=30)[:2]
        assert a[-30:] == b[:30]

    def test_whole_path_is_covered(self):
        path = self._path(200)
        covered = {p for seg, _ in split_path(path) for p in seg}
        assert covered == set(path)

    def test_trailing_stub_of_20_or_fewer_points_is_dropped(self):
        # a 20-point stub would start at 180 for a 200-point path
        starts = [offset for _, offset in split_path(self._path(200))]
        assert 180 not in starts

    def test_path_of_20_points_or_fewer_gives_no_segments(self):
        assert split_path(self._path(20)) == []

    def test_path_of_21_points_gives_one_segment(self):
        assert len(split_path(self._path(21))) == 1

    def test_custom_segment_length_and_overlap(self):
        segments = split_path(self._path(100), segment_length=50, overlap=10)
        assert [offset for _, offset in segments] == [0, 40, 80][:len(segments)]
        assert all(len(seg) <= 50 for seg, _ in segments)


#***************************************************************
# extract_full_stenosis_region
#***************************************************************

class TestExtractRegion:
    def test_region_brackets_the_narrowest_point(self):
        left, right = extract_full_stenosis_region(50, DIP)
        assert left < 50 < right

    def test_region_expands_until_the_vessel_has_recovered(self):
        left, right = extract_full_stenosis_region(50, DIP, recovery_ratio=0.8)
        target = 0.8 * 10.0
        assert DIP[left] >= target
        assert DIP[right] >= target

    def test_is_symmetric_for_a_symmetric_dip(self):
        left, right = extract_full_stenosis_region(50, DIP)
        assert 50 - left == right - 50

    def test_expansion_is_limited_by_max_expand(self):
        wide = dip_profile(half_width=40)
        left, right = extract_full_stenosis_region(50, wide, max_expand=5)
        assert left >= 50 - 5
        assert right <= 50 + 5

    def test_flat_profile_falls_back_to_the_minimum_size(self):
        left, right = extract_full_stenosis_region(50, FLAT)
        assert (left, right) == (42, 58)

    def test_index_past_the_end_is_returned_unchanged(self):
        assert extract_full_stenosis_region(500, DIP) == (500, 500)

    def test_index_at_the_start_is_clamped_into_the_profile(self):
        left, right = extract_full_stenosis_region(0, DIP)
        assert left == 0
        assert 0 < right <= len(DIP) - 1

    def test_index_at_the_end_is_clamped_into_the_profile(self):
        left, right = extract_full_stenosis_region(len(DIP) - 1, DIP)
        assert right == len(DIP) - 1
        assert 0 <= left < len(DIP) - 1


#***************************************************************
# build_exact_stenosis_mask
#***************************************************************

class TestBuildStenosisMask:
    @staticmethod
    def _band_mask():
        mask = np.zeros((40, 100), np.uint8)
        mask[10:31, :] = 255  # vessel, 21 px tall
        return mask

    @staticmethod
    def _pts(x0=40, x1=50, y=20):
        return [(x, y) for x in range(x0, x1 + 1)]

    def test_output_is_a_uint8_binary_image_of_the_same_shape(self):
        mask = self._band_mask()
        out = build_exact_stenosis_mask(self._pts(), mask)
        assert out.shape == mask.shape
        assert out.dtype == np.uint8
        assert set(np.unique(out)) <= {0, 255}

    def test_region_never_leaves_the_vessel(self):
        mask = self._band_mask()
        out = build_exact_stenosis_mask(self._pts(), mask)
        assert ((out > 0) <= (mask > 0)).all()

    def test_points_outside_the_vessel_are_discarded(self):
        mask = self._band_mask()
        out = build_exact_stenosis_mask(self._pts() + [(5, 2)], mask)
        assert out[2, 5] == 0

    def test_region_contains_the_original_points(self):
        mask = self._band_mask()
        pts = self._pts()
        out = build_exact_stenosis_mask(pts, mask)
        assert all(out[y, x] == 255 for x, y in pts)

    def test_region_grows_to_the_vessel_walls(self):
        mask = self._band_mask()
        out = build_exact_stenosis_mask(self._pts(), mask)
        assert (out[10:31, 45] == 255).all()

    def test_growth_is_limited_to_twelve_pixels(self):
        mask = self._band_mask()
        out = build_exact_stenosis_mask(self._pts(40, 50), mask)
        xs = np.where(out.any(axis=0))[0]
        assert xs.min() >= 40 - 12
        assert xs.max() <= 50 + 12

    def test_no_points_gives_an_empty_mask(self):
        out = build_exact_stenosis_mask([], self._band_mask())
        assert not out.any()


#***************************************************************
# get_main_contour
#***************************************************************

class TestGetMainContour:
    def test_empty_mask_gives_none(self):
        assert get_main_contour(np.zeros((20, 20), np.uint8)) is None

    def test_single_blob(self):
        region = np.zeros((50, 50), np.uint8)
        region[10:20, 10:30] = 255
        contour = get_main_contour(region)
        assert cv2.boundingRect(contour) == (10, 10, 20, 10)

    def test_largest_of_several_blobs_is_returned(self):
        region = np.zeros((60, 100), np.uint8)
        region[5:10, 5:10] = 255     # small
        region[20:50, 40:90] = 255   # large
        contour = get_main_contour(region)
        assert cv2.boundingRect(contour) == (40, 20, 50, 30)


#***************************************************************
# contours_close_or_overlap
#***************************************************************

class TestContoursCloseOrOverlap:
    def test_overlapping_rectangles(self):
        a = rect_contour(10, 10, 20, 20)
        b = rect_contour(20, 20, 20, 20)
        assert contours_close_or_overlap(a, b)

    def test_touching_rectangles(self):
        a = rect_contour(10, 10, 10, 10)
        b = rect_contour(19, 10, 10, 10)
        assert contours_close_or_overlap(a, b, distance_threshold=0)

    def test_close_rectangles_are_merged_within_the_threshold(self):
        a = rect_contour(0, 10, 10, 10)    # x 0..9
        b = rect_contour(25, 10, 10, 10)   # x 25..34, gap of 15 px
        assert contours_close_or_overlap(a, b, distance_threshold=20)

    def test_far_rectangles_are_not_merged(self):
        a = rect_contour(0, 10, 10, 10)
        b = rect_contour(40, 10, 10, 10)   # gap of 30 px
        assert not contours_close_or_overlap(a, b, distance_threshold=20)

    def test_threshold_is_respected(self):
        a = rect_contour(0, 10, 10, 10)
        b = rect_contour(25, 10, 10, 10)
        assert not contours_close_or_overlap(a, b, distance_threshold=5)

    def test_diagonal_distance_uses_both_axes(self):
        a = rect_contour(0, 0, 10, 10)
        b = rect_contour(25, 25, 10, 10)   # dx = dy = 15 -> distance ~21
        assert not contours_close_or_overlap(a, b, distance_threshold=20)
        assert contours_close_or_overlap(a, b, distance_threshold=22)

    def test_is_symmetric(self):
        a = rect_contour(0, 10, 10, 10)
        b = rect_contour(25, 10, 10, 10)
        assert contours_close_or_overlap(a, b) == contours_close_or_overlap(b, a)


#***************************************************************
# merge_contours
#***************************************************************

class TestMergeContours:
    def test_overlapping_contours_become_one_larger_contour(self):
        vessel = np.full((120, 240), 255, np.uint8)
        a = rect_contour(10, 40, 30, 20)
        b = rect_contour(30, 40, 30, 20)
        merged = merge_contours(a, b, vessel)
        x, y, w, h = cv2.boundingRect(merged)
        assert x <= 10 and x + w >= 60

    def test_merged_area_is_at_least_as_big_as_either_input(self):
        vessel = np.full((120, 240), 255, np.uint8)
        a = rect_contour(10, 40, 30, 20)
        b = rect_contour(30, 40, 30, 20)
        merged = merge_contours(a, b, vessel)
        assert cv2.contourArea(merged) >= max(cv2.contourArea(a), cv2.contourArea(b))

    def test_result_stays_inside_the_vessel(self):
        vessel = np.zeros((120, 240), np.uint8)
        vessel[40:60, :] = 255
        a = rect_contour(10, 30, 30, 40)   # sticks out of the vessel
        b = rect_contour(30, 30, 30, 40)
        merged = merge_contours(a, b, vessel)
        _, y, _, h = cv2.boundingRect(merged)
        assert y >= 40 and y + h <= 60

    def test_falls_back_to_the_first_contour_when_nothing_is_left(self):
        empty_vessel = np.zeros((120, 240), np.uint8)
        a = rect_contour(10, 40, 30, 20)
        b = rect_contour(30, 40, 30, 20)
        assert merge_contours(a, b, empty_vessel) is a

    @pytest.mark.xfail(
        reason="merge_contours only dilates by 1 px before taking the largest "
               "connected component, so two contours separated by a gap "
               "(contours_close_or_overlap allows up to 20 px) are not "
               "actually joined: one of the two lesions is silently dropped.",
        strict=False,
    )
    def test_nearby_but_separate_contours_are_both_kept(self):
        vessel = np.full((120, 240), 255, np.uint8)
        a = rect_contour(10, 40, 20, 20)   # x 10..29
        b = rect_contour(40, 40, 20, 20)   # x 40..59, gap of 10 px
        assert contours_close_or_overlap(a, b)

        merged = merge_contours(a, b, vessel)
        x, _, w, _ = cv2.boundingRect(merged)
        assert x <= 10 and x + w >= 60
