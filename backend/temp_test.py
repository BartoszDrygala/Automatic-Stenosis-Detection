import cv2
from pathlib import Path

from stenosis_pipeline.pipeline import find_stenosis

MASK_PATH = Path(__file__).resolve().parent.parent / "data" / "masks" / "1.png"

mask = cv2.imread(
    str(MASK_PATH),
    cv2.IMREAD_GRAYSCALE,
)

if mask is None:
    raise FileNotFoundError(f"Could not load mask: {MASK_PATH}")

detections = find_stenosis(mask)

print(f"Number of detections: {len(detections)}")
print(detections[0].keys() if detections else "No detections")