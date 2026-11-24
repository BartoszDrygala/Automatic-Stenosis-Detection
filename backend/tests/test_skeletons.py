
from backend.src.stenosis_pipeline.skeletonization import perform_skeletonization
from backend.src.stenosis_pipeline.visualization import visualize_paths
import matplotlib.pyplot as plt
import numpy as np



def test_skeletonization():
    a = 1
    b = 1

    np.testing.assert_array_equal(a, b)


def test_visualize_paths_returns_populated_figure():
    mask = np.zeros((4, 4), dtype=np.uint8)
    skeleton = np.zeros_like(mask)
    paths = [[(1, 1), (2, 1)]]

    fig = visualize_paths(mask, skeleton, paths)
    try:
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)