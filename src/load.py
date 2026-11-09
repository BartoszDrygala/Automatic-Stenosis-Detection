import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plots import plot_stenosis

def load_annotations(input_path:str | Path) -> list:
    with open(input_path, "r") as file:
        annotations = json.load(file)

    detections = annotations["detections"]

    for det in detections:
        det["path"] = np.asarray(det["path"])
        det["contour"] = np.asarray(det["contour"])

    return detections


def main():

    project_root = Path(__file__).resolve().parent.parent
    number = 1
    INPUT_PATH = project_root / "data" / "output" / "annotations" / f"{number}.json"
    INPUT_MASK_PATH = project_root / "data" / "masks" / f"{number}.png"
    mask = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    det = load_annotations(INPUT_PATH)
    fig = plot_stenosis(mask, det)
    plt.show(block=True)

if __name__ == "__main__":
    main()
    