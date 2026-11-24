import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .visualization import plot_stenosis

def load_annotations(input_path:str | Path) -> list:
    with open(input_path, "r") as file:
        annotations = json.load(file)

    detections = annotations["detections"]

    for det in detections:
        det["path"] = np.asarray(det["path"])
        det["contour"] = np.asarray(det["contour"])

    return detections


def main():
    print("LOAD")

if __name__ == "__main__":
    main()
    