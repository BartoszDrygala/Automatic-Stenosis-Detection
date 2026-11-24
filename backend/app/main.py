from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title = "stenosis detection API",
    description = "API for automatic stenosis detection in coronary angiography images.",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return{"status": "ok"}

app.include_router(router)