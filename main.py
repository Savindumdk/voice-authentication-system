import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging early.
from logging_config import configure_logging

logger = configure_logging()

# Import our custom modules
from database import initialize_database, close_database_connection

# Import global models (this will auto-load them)
from models import speaker_encoder, speaker_verifier, model_manager, DEVICE, CONFIG

# Import router after models are loaded
from router import router


# ----------------------------
# FastAPI Application
# ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (replaces deprecated @app.on_event)."""
    db_ok = initialize_database()
    app.state.db_ready = bool(db_ok)
    app.state.models_ready = speaker_encoder is not None and speaker_verifier is not None
    logger.info(
        "Startup complete | device=%s | db_ready=%s | models_ready=%s",
        DEVICE, app.state.db_ready, app.state.models_ready,
    )
    yield
    close_database_connection()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Voice Authentication API 🛡️",
    description="A secure pipeline for voice identification.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration.
# Origins come from the ALLOWED_ORIGINS env var (comma-separated). A wildcard
# "*" combined with allow_credentials=True is invalid and unsafe, so we default
# to localhost only. Set ALLOWED_ORIGINS to your real frontend origin(s) in prod.
_allowed_origins = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # Also allow any local dev origin (localhost / 127.0.0.1 on any port, e.g. a
    # Live Server tab or the page opened on a different port). This is safe —
    # localhost origins only come from the same machine. For production, serve
    # the frontend from a real domain and add it via ALLOWED_ORIGINS.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
print(f"🔒 CORS allowed origins: {_allowed_origins} (+ localhost/127.0.0.1 any port)")

# Include router
app.include_router(router)

# Mount static files for the web interface
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/")
async def serve_web_interface():
    """Serve the web interface."""
    return FileResponse("enhanced_web.html")


@app.get("/health", tags=["Ops"])
async def health():
    """Liveness probe — process is up. Always cheap, no dependencies."""
    return {"status": "ok"}


@app.get("/ready", tags=["Ops"])
async def ready():
    """Readiness probe — models and DB are available to serve traffic."""
    models_ready = speaker_encoder is not None and speaker_verifier is not None
    db_ready = getattr(app.state, "db_ready", False)
    ready = models_ready and db_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "models_ready": models_ready,
            "db_ready": db_ready,
            "device": str(DEVICE),
        },
    )


if __name__ == "__main__":
    print("🚀 Starting Voice Authentication API...")
    print(f"🎮 Using device: {DEVICE}")
    print("📡 All models loaded and ready!")
    print("🌐 Web interface available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    
    # Run the application
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
 