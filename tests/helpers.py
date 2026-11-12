"""Synthetic data builders shared by the test modules."""
import cv2
import numpy as np

from skeletonize import perform_skeletonization
from skeletons_postprocessing import perform_skeleton_postprocessing
from stenosis import detect_all_stenosis

# Vessel geometry used by make_vessel_mask. The skeleton of the generated
# vessel starts at x = MARGIN + 10 (skeletonization trims the ends) and the
# centre line sits at y = HEIGHT // 2.
MARGIN = 20
HEIGHT = 100


def make_vessel_mask(centers=(), length=300, half=10, narrow_half=3,
                     sigma=6, height=HEIGHT, margin=MARGIN):
    """
    Horizontal vessel of half-width `half` pixels with Gaussian narrowings.

    Each value in `centers` is an offset (in pixels, from the start of the
    vessel) where the half-width dips towards `narrow_half`. A narrowing
    centred at offset c ends up at image x = margin + c.

    With the defaults the healthy distance-transform value is 11 and the
    narrowest point is 4, i.e. a stenosis severity of about 0.64.
    """
    mask = np.zeros((height, length + 2 * margin), np.uint8)
    y0 = height // 2

    for x in range(margin, margin + length):
        hw = float(half)
        for c in centers:
            dip = (half - narrow_half) * np.exp(-((x - margin - c) ** 2) / (2 * sigma ** 2))
            hw = min(hw, half - dip)
        hw = int(round(hw))
        mask[y0 - hw:y0 + hw + 1, x] = 255

    return mask


def run_detection(mask):
    """Skeleton -> paths -> distance transform -> detections."""
    skeleton = perform_skeletonization(mask)
    paths = perform_skeleton_postprocessing(skeleton)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    detections = detect_all_stenosis(paths, dist, mask)
    return skeleton, paths, dist, detections


def dip_profile(n=100, center=50, base=10.0, depth=7.0, half_width=10):
    """
    1D vessel-radius profile: constant `base` with a V-shaped dip.

    At `center` the value is base - depth; it recovers linearly to `base`
    over `half_width` samples on each side.
    """
    i = np.arange(n, dtype=float)
    return base - depth * np.clip(1 - np.abs(i - center) / half_width, 0, None)


def profile_to_path_dist(profile):
    """
    Turn a 1D profile into the (path, dist) pair the detectors expect:
    a horizontal path of (x, y) points over a 1-row distance 'image'.
    """
    profile = np.asarray(profile, dtype=float)
    dist = profile.reshape(1, -1)
    path = [(x, 0) for x in range(len(profile))]
    return path, dist


def rect_contour(x, y, w, h, shape=(120, 240)):
    """A real OpenCV contour of a filled w x h rectangle at (x, y)."""
    canvas = np.zeros(shape, np.uint8)
    canvas[y:y + h, x:x + w] = 255
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]
