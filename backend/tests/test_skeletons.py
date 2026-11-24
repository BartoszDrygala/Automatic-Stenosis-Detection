
from src.skeletonize import perform_skeletonization
import cv2
import numpy as np

mask = cv2.imread("data/masks/1.png", cv2.IMREAD_GRAYSCALE)

def test_skeletonization():
    a = perform_skeletonization(mask)
    b = perform_skeletonization(mask)

    np.testing.assert_array_equal(a, b)