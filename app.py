# app.py
# FastAPI web backend — exposes the prediction model as an HTTP API endpoint.
#
# This is the bridge between your web/mobile frontend and the ML model.
# The frontend sends a photo, this server runs the prediction, and returns JSON.
#
# To run the server:
#   pip install fastapi uvicorn python-multipart
#   uvicorn app:app --reload
#
# Then open http://localhost:8000/docs in your browser to test the API interactively.

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse   # serves static files (our HTML page)
from fastapi.staticfiles import StaticFiles  # mounts the static/ folder

# Import the predict function from our prediction module
from predict import predict

# ─── App setup ────────────────────────────────────────────────────────────────

# Create the FastAPI application instance
app = FastAPI(
    title="Maize Disease Detection API",
    description="Upload a maize leaf photo to detect crop diseases.",
    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing) — allows your frontend (React, Next.js, etc.)
# running on a different port/domain to call this API without being blocked by the browser.
# In production, replace "*" with your actual frontend domain for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow requests from any origin (dev only)
    allow_methods=["GET", "POST"], # GET serves the HTML page; POST submits images
    allow_headers=["*"],
)

# Mount the static/ folder so the server can deliver index.html and any future
# CSS/JS/image assets. Files are served at /static/... URLs.
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Allowed file types ───────────────────────────────────────────────────────

# Only accept common image formats that PIL can open
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Health check endpoint ────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the HTML frontend. Opening http://localhost:8000 loads the UI."""
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    """JSON health check — useful for automated monitoring."""
    return {"status": "ok", "message": "Maize Disease Detection API is running."}


# ─── Prediction endpoint ──────────────────────────────────────────────────────

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Accept an image upload and return a disease prediction.

    Request:  POST /predict  with a multipart form field named 'file'
    Response: JSON with class, confidence, description, and all_scores

    Example response:
    {
        "class": "Common_Rust",
        "confidence": 87.4,
        "description": "Powdery pustules on both leaf surfaces...",
        "all_scores": {
            "Healthy": 2.1,
            "Gray_Leaf_Spot": 5.2,
            "Blight": 5.3,
            "Common_Rust": 87.4
        }
    }
    """
    # Validate the uploaded file has an accepted image extension
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a JPG or PNG image.",
        )

    # Save the uploaded bytes to a temporary file on disk.
    # The model needs a file path, not raw bytes.
    # tempfile.NamedTemporaryFile creates a file that auto-deletes when closed.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        contents = await file.read()   # read all uploaded bytes
        tmp.write(contents)            # write them to a temp file
        tmp_path = tmp.name            # remember the temp file path

    try:
        # Run the prediction using the core predict() function
        result = predict(tmp_path)
    finally:
        # Always clean up the temp file, even if an error occurs
        os.unlink(tmp_path)

    return result
