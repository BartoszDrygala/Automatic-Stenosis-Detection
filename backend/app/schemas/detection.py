from pydantic import BaseModel, Field

class Detection(BaseModel):
    id: int
    severity: float = Field(ge = 0, le = 1, description = 'Severity in a fraction')
    contour: list[tuple[int, int]]

class DetectionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detection_count: int
    detections: list[Detection]
    visualization: str
