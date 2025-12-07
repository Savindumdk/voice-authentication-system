import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from database import initialize_database, close_database_connection

# Import global models (this will auto-load them)
from models import speaker_encoder, speaker_verifier, model_manager, DEVICE, CONFIG

# Import router after models are loaded
from router import router

# ----------------------------
# FastAPI Application
# ----------------------------

# Initialize database connection
initialize_database()

app = FastAPI(
    title="Voice Authentication API 🛡️",
    description="A secure pipeline for voice identification.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

# Mount static files for the web interface
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/")
async def serve_web_interface():
    """Serve the web interface."""
    return FileResponse("enhanced_web.html")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    close_database_connection()

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
