import cv2
import numpy as np

from skeletonize import perform_skeletonization


def _band(h=40, w=100, y0=10, y1=31, x0=5, x1=95):
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _n_components(binary):
    return cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=8)[0] - 1


def test_output_is_uint8_with_only_0_and_255():
    skel = perform_skeletonization(_band())
    assert skel.dtype == np.uint8
    assert set(np.unique(skel)) <= {0, 255}


def test_shape_is_preserved():
    mask = _band()
    assert perform_skeletonization(mask).shape == mask.shape


def test_empty_mask_gives_empty_skeleton():
    skel = perform_skeletonization(np.zeros((20, 20), np.uint8))
    assert not skel.any()


def test_skeleton_lies_inside_the_mask():
    mask = _band()
    skel = perform_skeletonization(mask)
    assert skel.any()
    assert ((skel > 0) <= (mask > 0)).all()


def test_skeleton_is_much_thinner_than_the_mask():
    mask = _band()
    skel = perform_skeletonization(mask)
    assert (skel > 0).sum() < 0.1 * (mask > 0).sum()


def test_single_vessel_gives_a_single_connected_skeleton():
    assert _n_components(perform_skeletonization(_band())) == 1


def test_two_separate_vessels_give_two_components():
    mask = np.zeros((80, 100), np.uint8)
    mask[10:20, 5:95] = 255
    mask[50:60, 5:95] = 255
    assert _n_components(perform_skeletonization(mask)) == 2


def test_any_positive_value_counts_as_foreground():
    """The mask is binarised with `> 0`, so 1, 128 and 255 are equivalent."""
    ref = perform_skeletonization(_band())
    for value in (1, 128):
        mask = (_band() > 0).astype(np.uint8) * value
        assert np.array_equal(perform_skeletonization(mask), ref)


def test_input_mask_is_not_modified():
    mask = _band()
    before = mask.copy()
    perform_skeletonization(mask)
    assert np.array_equal(mask, before)


def test_one_pixel_wide_line_is_preserved():
    mask = np.zeros((20, 50), np.uint8)
    mask[10, 5:45] = 255
    skel = perform_skeletonization(mask)
    assert np.array_equal(skel, mask)
