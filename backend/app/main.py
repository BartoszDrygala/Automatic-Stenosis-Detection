
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router


FRONTEND_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "frontend"
    / "index.html"
)

app = FastAPI(
    title="Stenosis Detection API",
    description="API for automatic detection of stenosis in coronary angiography images.",
    version="1.0.0"
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(FRONTEND_FILE, media_type="text/html")
