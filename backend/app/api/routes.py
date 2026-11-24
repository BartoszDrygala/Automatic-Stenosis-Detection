import numpy as np
import cv2

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.detection import Detection, DetectionResponse
from app.services.detection_service import process_mask
from app.services.visualization_service import create_visualization

router = APIRouter(
    prefix="/api/v1",
    tags=["detections"]
    
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

@router.post("/detections", response_model=DetectionResponse)
async def detect_stenosis(file:UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file is too large."
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    image_decoded = np.frombuffer(content, dtype=np.uint8)
    mask = cv2.imdecode(image_decoded, cv2.IMREAD_GRAYSCALE)
    
    height, width = mask.shape
    detections = await run_in_threadpool(process_mask, mask)

    visualization = await run_in_threadpool(create_visualization, mask, detections)

    results = [Detection(
        id = i,
        severity=float(det["severity"]),
        contour = [(int(x), int(y)) for x, y in det["contour"].reshape(-1,2)]
        ) 
        for i, det in enumerate(detections)
    ]

    return DetectionResponse(
        filename=file.filename or "uploaded_image",
        image_width=width,
        image_height=height,
        detection_count=len(results),
        detections=results,
        visualization=visualization
    )