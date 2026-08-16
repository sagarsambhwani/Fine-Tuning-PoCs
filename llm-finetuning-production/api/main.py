"""
Production-Style FastAPI Serving Application
Exposes structured extraction inference via REST API.
"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ExtractionRequest, ExtractionResponse, HealthResponse, OrderDetails
from api.inference import initialize_model, get_inference_engine, shutdown_model, get_model_path
from src.utils.logger import setup_logger

logger = setup_logger("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: loads weights once on startup and cleans up on shutdown."""
    logger.info("Starting up FastAPI LLM Serving application...")
    try:
        initialize_model()
    except Exception as e:
        logger.warning(f"Could not preload model on startup (running in lazy-load/test mode): {e}")
    yield
    logger.info("Shutting down FastAPI LLM Serving application...")
    shutdown_model()

app = FastAPI(
    title="Fine-Tuned LLM Structured JSON Extractor API",
    description="Production-style serving API for fine-tuned Qwen-1.5B structured order extraction.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health and Readiness Probe"
)
async def health_check():
    """Returns service operational state, active model identifier, and compute device."""
    try:
        engine = get_inference_engine()
        model_name = str(engine.model_path_or_id)
        device = engine.device
    except Exception:
        model_name = get_model_path()
        device = "cpu"

    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device
    )

@app.post(
    "/predict",
    response_model=ExtractionResponse,
    tags=["Inference"],
    summary="Extract Structured Order JSON from Natural Language"
)
async def predict_order(request: ExtractionRequest):
    """
    Accepts unstructured natural language text describing an order, runs inference
    through the fine-tuned LLM, and returns validated structured JSON.
    """
    try:
        engine = get_inference_engine()
        result = engine.extract(
            text=request.text,
            temperature=request.temperature
        )

        parsed_data = None
        if result["is_valid_json"] and result["data"]:
            parsed_data = OrderDetails(**result["data"])

        return ExtractionResponse(
            data=parsed_data,
            raw_response=result["raw_response"],
            is_valid_json=result["is_valid_json"],
            error=result["error"],
            latency_ms=result["latency_ms"],
            model_name=result["model_name"]
        )

    except Exception as e:
        logger.error(f"Inference error during /predict: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
