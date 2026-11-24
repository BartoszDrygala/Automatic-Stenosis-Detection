import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def draw_stenosis(vis:np.ndarray, detections:list, draw_text:bool=True):
    """
    Draw detected stenosis contours and labels on an image/mask.

    Parameters
    ----------
    vis : np.ndarray
        RGB image or mask used for visualization.
    detections : list
        List of stenosis detections.
    draw_text : bool
        Whether to draw severity labels.

    Returns
    -------
    np.ndarray
        Visualization with stenosis detections drawn on it.
    """

    for det in detections:

        path = det["path"]
        idx = det["idx"]
        severity = det["severity"]
        contour = det["contour"]

        g_set = det["g_set"]
        l_set = det["l_set"]
        orig_idx = det["orig_idx"]

        if idx >= len(path):
            continue

        # Determine detection type
        if orig_idx in g_set and orig_idx in l_set:
            label = "B"
        elif orig_idx in g_set:
            label = "G"
        else:
            label = "L"

        x, y = path[idx]

        text = f"{label}:{int(severity * 100)}%"

        # Draw stenosis contour
        cv2.drawContours(vis, [contour], -1, (255, 0, 0), 1)

        # Draw label
        if draw_text:
            cv2.putText(vis, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)

    return vis


def plot_stenosis(mask:np.ndarray, detections:list, image:np.ndarray=None):
    """
    Plot detected stenosis on a mask, optionally also on the original image.

    Parameters
    ----------
    mask : np.ndarray
        Vessel mask.
    detections : list
        List of stenosis detections.
    image : np.ndarray, optional
        Original angiography image.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the visualization.
    """

    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Draw detections on mask
    mask_vis = draw_stenosis(mask_vis, detections)

    if image is not None:

        img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Draw detections on image
        img_vis = draw_stenosis(img_vis, detections)

        # Image + mask
        fig = plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img_vis)
        plt.title("Image + Stenosis")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_vis)
        plt.title("Mask + Stenosis")
        plt.axis("off")

    else:

        # Mask only
        fig = plt.figure(figsize=(5, 5))

        plt.imshow(mask_vis)
        plt.title("Detected Stenosis")
        plt.axis("off")

    plt.tight_layout()

    return fig


#used for overlaying the skeleton on top of the image
def plot_overlay(mask:np.ndarray, skeleton:np.ndarray, image:np.ndarray = None):
    """
    Plot overlay of skeleton on the mask and the image (optionally)

    Parameters
    ----------

    mask : np.ndarray
        Vessel mask
    skeleton : np.ndarray
        Skeleton of the vessel
    image : np.ndarray, optional
        Vessel image

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
    """
    skeleton = (skeleton > 0)
    if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[2] == 1):
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # If an image is provided, prepare both overlays and plot side-by-side
    if image is not None:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay_img = img_rgb.copy()
        overlay_mask = mask_rgb.copy()

        overlay_img[skeleton] = [255, 0, 0]
        overlay_mask[skeleton] = [255, 0, 0]

        fig = plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(overlay_mask)
        plt.title("Skeleton Overlay on Mask")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay_img)
        plt.title("Skeleton Overlay on Image")
        plt.axis("off")

        plt.tight_layout()
        return fig

    # No image: just show overlay on mask
    overlay_mask = mask_rgb.copy()
    overlay_mask[skeleton] = [255, 0, 0]

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(overlay_mask)
    plt.title("Skeleton Overlay on Mask")
    plt.axis("off")
    return fig

#used for visualizing separate paths in different colors
def visualize_paths(mask:np.ndarray, skeleton:np.ndarray, paths:list, image:np.ndarray = None):
    """
    Visualizing separate paths of the skeleton

    Parameters
    ----------
    mask : np.ndarray
        Vessel mask
    skeleton : np.ndarray
        Skeleton of the vessel
    paths : list
        List of subsequent paths of the skeleton
    image : np.ndarray
        Vessel image

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the visualization
    """
    if image is not None:
        plt.figure(figsize=(10,10))
        size_a = 2
        size_b = 2
        c = 1
        # Original image
        plt.subplot(size_a, size_b, c)
        plt.title("Original Image")
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    else:
        plt.figure(figsize=(10,3))
        size_a = 1
        size_b = 3
        c = 0

    # Mask
    plt.subplot(size_a, size_b, c + 1)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis("off")

    # Skeleton
    plt.subplot(size_a, size_b, c + 2)
    plt.title("Skeleton")
    plt.imshow(skeleton, cmap='gray')
    plt.axis("off")

    # Paths (colored)
    plt.subplot(size_a, size_b, c + 3)
    plt.title("Extracted Paths")

    canvas = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)

    for path in paths:
        color = [random.randint(50, 255) for _ in range(3)]

        for (x, y) in path:
            canvas[y, x] = color  #reverse order cuz row, column i think

    plt.imshow(canvas)
    plt.axis("off")

    plt.tight_layout()
    return plt