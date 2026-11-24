import numpy as np
from skimage.morphology import skeletonize


def perform_skeletonization(mask):
    """
    Skeletonization of the given mask

    Parameters
    ----------
    mask : np.ndarray
        Vessel mask

    Returns
    -------
    skeleton_img : np.ndarray
        Skeleton image
    """
    #binarizing mask if it wasn't already done
    binary_mask = (mask > 0).astype(np.uint8)

    #main skeletonization
    skeleton = skeletonize(binary_mask)

    #givin higher val for readability
    skeleton_img = (skeleton * 255).astype(np.uint8)

    return skeleton_img