import os
import json
import numpy as np

# Self made Encoder that supports numpy types, cuz normal built-in .json encoder doesn't
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (set, frozenset)):
            return sorted(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)
        
        return super().default(obj)


def save_annotations(PATH:str | os.PathLike[str], detections:list, image_name:str | int) -> None:
    """
    Saving found stenosis regions into .json file.

    Parameters
    ----------
    PATH : str | os.PathLike[str]
        A path to an output directory
    detections : list
        List of detected stenosis regions
    image_name : str | int
        Name or identifier of the image

    Returns
    -------
    None
    """

    output = {
        "image_id": image_name,
        "detections": []
    }

    for det in detections:
        saved_detection = {
            "path": det["path"],
            "idx": det["idx"],
            "orig_idx": det["orig_idx"],
            "severity": det["severity"],
            "contour": det["contour"],
            "left": det["left"],
            "right": det["right"],
            "g_set": det["g_set"],
            "l_set": det["l_set"]
        }

        output["detections"].append(saved_detection)

    serialized_output = json.dumps(output, indent=4, cls=NumpyEncoder)
    with open(PATH, "w") as file:
        file.write(serialized_output)