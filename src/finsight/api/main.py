import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finsight.api.routes import router as chat_router

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="FinSight API",
    description="Agentic RAG for Financial Intelligence",
    version="1.0.0"
)

# Configure CORS for the frontend (Vercel / Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat_router)

@app.get("/health", tags=["system"])
async def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "ok", "service": "FinSight API"}

if __name__ == "__main__":
    import uvicorn
    # This allows running the file directly for testing
    logger.info("Starting FinSight API on port 8000...")
    uvicorn.run("finsight.api.main:app", host="0.0.0.0", port=8000, reload=True)
