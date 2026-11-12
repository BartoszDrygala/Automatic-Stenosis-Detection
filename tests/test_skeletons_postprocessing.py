import numpy as np
import pytest

from skeletonize import perform_skeletonization
from skeletons_postprocessing import (
    detecting_nodes,
    edges,
    get_neighbours,
    get_paths,
    keep_main_vessels,
    perform_skeleton_postprocessing,
    prune_short_branches,
    remove_bifurcation_points,
)


def _line(n=20, y=10, x0=5, shape=(30, 40)):
    skel = np.zeros(shape, np.uint8)
    skel[y, x0:x0 + n] = 255
    return skel


def _t_shape():
    skel = np.zeros((30, 30), np.uint8)
    skel[10, 0:21] = 255   # horizontal bar
    skel[10:21, 10] = 255  # vertical stem
    return skel


#***************************************************************
# get_neighbours
#***************************************************************

class TestGetNeighbours:
    def test_interior_pixel_of_full_image_has_eight_neighbours(self):
        skel = np.ones((5, 5), np.uint8)
        assert len(get_neighbours(2, 2, skel)) == 8

    def test_pixel_itself_is_not_a_neighbour(self):
        skel = np.ones((5, 5), np.uint8)
        assert (2, 2) not in get_neighbours(2, 2, skel)

    def test_corner_only_sees_pixels_inside_the_image(self):
        skel = np.ones((5, 5), np.uint8)
        assert sorted(get_neighbours(0, 0, skel)) == [(0, 1), (1, 0), (1, 1)]

    def test_far_corner_does_not_wrap_around(self):
        skel = np.ones((5, 5), np.uint8)
        assert sorted(get_neighbours(4, 4, skel)) == [(3, 3), (3, 4), (4, 3)]

    def test_only_skeleton_pixels_are_returned(self):
        skel = np.zeros((5, 5), np.uint8)
        skel[2, 2] = skel[2, 3] = skel[3, 3] = 255
        # (x, y) = (3, 2) and (3, 3) are the two other skeleton pixels
        assert set(get_neighbours(2, 2, skel)) == {(3, 2), (3, 3)}

    def test_returns_x_y_order(self):
        """Points are (x, y) = (column, row), while indexing is skel[y, x]."""
        skel = np.zeros((10, 10), np.uint8)
        skel[2, 7] = 255  # row 2, column 7
        assert get_neighbours(6, 2, skel) == [(7, 2)]


#***************************************************************
# detecting_nodes
#***************************************************************

class TestDetectingNodes:
    def test_empty_skeleton_has_no_nodes(self):
        assert detecting_nodes(np.zeros((10, 10), np.uint8)) == set()

    def test_straight_line_has_exactly_its_two_endpoints(self):
        assert detecting_nodes(_line()) == {(5, 10), (24, 10)}

    def test_single_pixel_is_a_node(self):
        skel = np.zeros((5, 5), np.uint8)
        skel[2, 2] = 255
        assert detecting_nodes(skel) == {(2, 2)}

    def test_t_junction_contains_endpoints_and_junction(self):
        nodes = detecting_nodes(_t_shape())
        assert {(0, 10), (20, 10), (10, 20)} <= nodes  # three endpoints
        assert (10, 10) in nodes                       # the junction

    def test_pixels_in_the_middle_of_a_line_are_not_nodes(self):
        nodes = detecting_nodes(_line())
        assert (12, 10) not in nodes


#***************************************************************
# edges
#***************************************************************

class TestEdges:
    def test_walks_from_one_endpoint_to_the_other(self):
        skel = _line()
        nodes = detecting_nodes(skel)
        path = edges((5, 10), (6, 10), skel, nodes)
        assert path[0] == (5, 10)
        assert path[-1] == (24, 10)
        assert len(path) == 20

    def test_path_is_contiguous(self):
        skel = _line()
        path = edges((5, 10), (6, 10), skel, detecting_nodes(skel))
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            assert max(abs(x1 - x2), abs(y1 - y2)) == 1

    def test_stops_at_a_dead_end(self):
        skel = np.zeros((5, 5), np.uint8)
        skel[0, 0] = skel[0, 1] = 255
        assert edges((0, 0), (1, 0), skel, {(0, 0), (1, 0)}) == [(0, 0), (1, 0)]

    def test_stops_at_a_junction_node(self):
        skel = _t_shape()
        nodes = detecting_nodes(skel)
        path = edges((0, 10), (1, 10), skel, nodes)
        assert path[-1] in nodes
        assert all(p not in nodes for p in path[1:-1])


#***************************************************************
# remove_bifurcation_points
#***************************************************************

class TestRemoveBifurcationPoints:
    def test_endpoints_are_removed_from_a_line(self):
        skel = _line()
        path = [(x, 10) for x in range(5, 25)]
        cleaned = remove_bifurcation_points(path, skel)
        assert cleaned == [(x, 10) for x in range(6, 24)]

    def test_junction_pixels_are_removed(self):
        skel = _t_shape()
        cleaned = remove_bifurcation_points([(10, 10)], skel)
        assert cleaned == []

    def test_empty_path(self):
        assert remove_bifurcation_points([], _line()) == []


#***************************************************************
# get_paths
#***************************************************************

class TestGetPaths:
    def test_empty_skeleton_has_no_paths(self):
        assert get_paths(np.zeros((10, 10), np.uint8)) == []

    def test_line_gives_at_least_one_path(self):
        assert len(get_paths(_line())) >= 1

    def test_paths_only_contain_skeleton_pixels(self):
        skel = _line()
        for path in get_paths(skel):
            assert all(skel[y, x] for x, y in path)

    def test_paths_exclude_endpoints_and_are_longer_than_five(self):
        skel = _line()
        nodes = detecting_nodes(skel)
        for path in get_paths(skel):
            assert len(path) > 5
            assert not nodes & set(path)

    def test_segments_that_are_too_short_are_dropped(self):
        # 6 pixels -> 4 after the endpoints are removed -> not > 5
        assert get_paths(_line(n=6)) == []

    def test_every_interior_pixel_of_a_line_is_covered(self):
        covered = {p for path in get_paths(_line()) for p in path}
        assert covered == {(x, 10) for x in range(6, 24)}

    @pytest.mark.xfail(
        reason="get_paths marks pixels as visited using the *cleaned* path "
               "(endpoints stripped), so the walk starting from the other "
               "endpoint isn't blocked and every segment is returned twice, "
               "once in each direction.",
        strict=False,
    )
    def test_each_segment_is_returned_only_once(self):
        paths = get_paths(_line())
        assert len(paths) == 1


#***************************************************************
# prune_short_branches
#***************************************************************

class TestPruneShortBranches:
    @staticmethod
    def _line_with_spur():
        skel = np.zeros((30, 60), np.uint8)
        skel[10, 5:55] = 255   # main vessel, 50 px
        skel[11:14, 30] = 255  # 3 px spur hanging off the middle
        return skel

    def test_short_spur_is_removed(self):
        pruned = prune_short_branches(self._line_with_spur(), 10)
        assert not pruned[11:14, 30].any()

    def test_main_vessel_is_kept_when_a_spur_is_removed(self):
        pruned = prune_short_branches(self._line_with_spur(), 10)
        assert (pruned[10] > 0).sum() == 50

    def test_isolated_short_segment_is_removed_entirely(self):
        skel = np.zeros((30, 40), np.uint8)
        skel[10, 5:10] = 255
        assert not prune_short_branches(skel, 10).any()

    def test_long_line_is_left_untouched(self):
        skel = _line()
        assert np.array_equal(prune_short_branches(skel, 10), skel)

    def test_min_length_controls_what_counts_as_short(self):
        skel = np.zeros((30, 40), np.uint8)
        skel[10, 5:10] = 255
        assert prune_short_branches(skel, 3).any()

    def test_input_is_not_modified(self):
        skel = self._line_with_spur()
        before = skel.copy()
        prune_short_branches(skel, 10)
        assert np.array_equal(skel, before)

    def test_empty_skeleton(self):
        assert not prune_short_branches(np.zeros((10, 10), np.uint8)).any()


#***************************************************************
# keep_main_vessels
#***************************************************************

class TestKeepMainVessels:
    @staticmethod
    def _two_vessels():
        mask = np.zeros((80, 120), np.uint8)
        mask[10:30, 5:115] = 255  # thick vessel (20 px)
        mask[50:53, 5:115] = 255  # thin vessel (3 px)
        return mask

    def test_thin_vessel_is_removed_and_thick_vessel_kept(self):
        mask = self._two_vessels()
        skel = perform_skeletonization(mask)
        kept = keep_main_vessels(skel, mask, min_ratio=0.5)

        assert (skel[50:53] > 0).any()          # thin one was there before
        assert not kept[50:53].any()            # ...and is gone now
        assert np.array_equal(kept[10:30] > 0, skel[10:30] > 0)

    def test_empty_skeleton_stays_empty(self):
        mask = self._two_vessels()
        assert not keep_main_vessels(np.zeros_like(mask), mask).any()

    def test_input_skeleton_is_not_modified(self):
        mask = self._two_vessels()
        skel = perform_skeletonization(mask)
        before = skel.copy()
        keep_main_vessels(skel, mask)
        assert np.array_equal(skel, before)

    def test_min_ratio_zero_keeps_everything(self):
        mask = self._two_vessels()
        skel = perform_skeletonization(mask)
        kept = keep_main_vessels(skel, mask, min_ratio=0.0)
        assert np.array_equal(kept, skel)


#***************************************************************
# perform_skeleton_postprocessing
#***************************************************************

class TestPostprocessingPipeline:
    def test_returns_list_of_lists_of_int_points(self):
        paths = perform_skeleton_postprocessing(_line())
        assert isinstance(paths, list) and paths
        for path in paths:
            assert isinstance(path, list)
            for x, y in path:
                assert isinstance(x, int) and isinstance(y, int)

    def test_empty_skeleton_gives_no_paths(self):
        assert perform_skeleton_postprocessing(np.zeros((10, 10), np.uint8)) == []

    def test_short_spurs_do_not_produce_their_own_paths(self):
        skel = np.zeros((30, 60), np.uint8)
        skel[10, 5:55] = 255
        skel[11:14, 30] = 255
        covered = {p for path in perform_skeleton_postprocessing(skel) for p in path}
        assert not covered & {(30, 11), (30, 12), (30, 13)}
