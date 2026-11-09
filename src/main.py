import cv2
import numpy as np
from pathlib import Path
from plots import plot_stenosis, plot_overlay, visualize_paths
from skeletonize import perform_skeletonization
from skeletons_postprocessing import perform_skeleton_postprocessing
from stenosis import detect_all_stenosis
from save import save_annotations



#***************************************************************
# MAIN PIPELINE:
# 1. Skeletonization - creating skeletons of each mask
# 2. Paths - creating fragments of the vessels between bifurcations or crossings
# 3. Distances - calculating distance transform for all the pixels of the mask
# 4. Detections - final detections of stenosis ready for plotting
#***************************************************************

def find_stenosis(mask:np.ndarray, image:np.ndarray = None):
    """
    Finding and marking stenosis regions and giving some utilities helpful for debugging

    Parameters
    ----------
    mask : np.ndarray
        Vessel mask
    image : np.ndarray, optional
        Vessel image

    Returns
    -------
    stenosis_image : np.ndarray
        Stenosis visualization
    overlay : np.ndarray
        Skeleton overlayed on the provided image
    paths_img : module
        Visualization of the paths
    """
    skeleton = perform_skeletonization(mask)
    paths = perform_skeleton_postprocessing(skeleton)
    distances = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    detections = detect_all_stenosis(paths, distances, mask)

    #if user gave original image and wants additional graphs
    if image is not None:
        stenosis_image = plot_stenosis(mask, detections, image)
        overlay = plot_overlay(image, skeleton, mask)
        paths_img = visualize_paths(mask, skeleton, paths, image)
    else:
        stenosis_image = plot_stenosis(mask, detections)
        overlay = plot_overlay(mask, skeleton)
        paths_img = visualize_paths(mask, skeleton, paths)
    
    # make it return all the plots also so its convinient for the user
    return stenosis_image, overlay, paths_img, detections


#***************************************************************
# Main entry of the program

# The function accepts following arguments:
# 1. mask - mask of the given vessel
# 2. image (optional) - original image for better visualization

# The function returns things You can use for visualization:
# 1. Image with stenosis
# 2. Detections - places where stenosis was detected
# 3. Paths - fragments of the vessels between the bifurcations

#***************************************************************
def main():
    number = 1
    name = f"{number}.png"
    project_root = Path(__file__).resolve().parent.parent

    INPUT_MASK_PATH = project_root / "data" / "masks" / name
    INPUT_IMAGE_PATH = project_root / "data" / "images" / name
    OUTPUT_PATH = project_root / "data" / "output"
    OUTPUT_IMG = OUTPUT_PATH / "graphics" / name
    OUTPUT_JSON = OUTPUT_PATH / "annotations" / f"{number}.json"

    #assure directoriy existance
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    mask = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if mask is None or image is None:
        raise FileNotFoundError(f"Could not read mask or image: {INPUT_MASK_PATH}, {INPUT_IMAGE_PATH}")

    stenosis, overlay, paths, detections = find_stenosis(mask, image)
    stenosis.show()
    overlay.show()
    paths.show()

    stenosis.savefig(OUTPUT_IMG)
    save_annotations(OUTPUT_JSON, detections, number)


if __name__ == "__main__":
    main()