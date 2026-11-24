from stenosis_pipeline.visualization import plot_stenosis

import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def create_visualization(mask, detections):
    fig = plot_stenosis(mask, detections)

    try:
        buffer = BytesIO()

        fig.savefig(buffer, format="png",  bbox_inches="tight", dpi=150)

        buffer.seek(0)

        image_bytes = buffer.getvalue()

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        return f"data:image/png;base64,{encoded_image}"
    
    finally:
        plt.close(fig)
