import numpy as np

from stenosis_pipeline.pipeline import find_stenosis

def process_mask(mask:np.ndarray):

    detections = find_stenosis(mask)

    return detections