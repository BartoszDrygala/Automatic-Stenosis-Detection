import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


def test_detection_response_matches_api_schema(monkeypatch):
    contour = np.array([[[0, 0]], [[1, 1]], [[1, 0]]], dtype=np.int32)
    monkeypatch.setattr(
        routes,
        "process_mask",
        lambda mask: [{"severity": 0.5, "contour": contour}],
    )
    monkeypatch.setattr(
        routes,
        "create_visualization",
        lambda mask, detections: "data:image/png;base64,",
    )

    success, encoded_image = cv2.imencode(".png", np.zeros((2, 3), dtype=np.uint8))
    assert success

    response = TestClient(app).post(
        "/api/v1/detections",
        files={"file": ("1.png", encoded_image.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "1.png",
        "image_width": 3,
        "image_height": 2,
        "detection_count": 1,
        "detections": [
            {
                "id": 0,
                "severity": 0.5,
                "contour": [[0, 0], [1, 1], [1, 0]],
            }
        ],
        "visualization": "data:image/png;base64,",
    }
