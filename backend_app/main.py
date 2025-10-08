"""
Car Troubleshooting Chatbot - FastAPI Backend
Main application entry point with CORS, middleware, and route registration.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend_app.api.chat import router as chat_router
from backend_app.api.diagnostics import router as diagnostics_router
from backend_app.core.settings import get_settings, reset_settings
from backend_app.services.rag_service import RAGService

# Global RAG service instance
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Car Troubleshooting Chatbot API")

    # Force reload settings BEFORE creating any services
    reset_settings()
    settings = get_settings()
    logger.info(f"🎯 Loaded similarity threshold: {settings.similarity_threshold}")

    # Initialize RAG service AFTER settings reset
    global rag_service
    rag_service = RAGService()
    await rag_service.initialize()

    # Store in app state for access in routes
    app.state.rag_service = rag_service

    logger.info("RAG service initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Car Troubleshooting Chatbot API")

    # Cleanup RAG service to prevent vector store corruption
    if rag_service:
        logger.info("Cleaning up RAG service...")
        try:
            await rag_service.cleanup()
        except Exception as e:
            logger.error(f"Error during RAG service cleanup: {e}")

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="European Car Troubleshooting Chatbot",
        description="AI-powered automotive diagnostics specialized for European markets",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# Add a simple GET handler for /api/chat to return a helpful message
@app.get("/api/chat")
async def chat_get_info():
    """GET endpoint for /api/chat that provides information about the chat API."""
    return {
        "message": "Chat API endpoint",
        "method": "POST",
        "description": "Use POST to send chat messages",
        "example": {"message": "BMW M3 issues", "vehicle_info": None, "country": "LT"},
    }


# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(diagnostics_router, prefix="/api", tags=["diagnostics"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "European Car Troubleshooting Chatbot API",
        "version": "1.0.0",
        "description": "AI-powered automotive diagnostics for European markets",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "running",
        "features": [
            "27+ European countries supported",
            "Lithuanian market specialized",
            "RAG-powered diagnostics",
            "Function calling for cost estimates",
            "Cold climate considerations",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if RAG service is initialized
        if not hasattr(app.state, "rag_service") or app.state.rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        return {
            "status": "healthy",
            "rag_service": "initialized",
            "timestamp": str(asyncio.get_event_loop().time()),
            "features": {
                "rag_retrieval": "active",
                "function_calling": "active",
                "european_knowledge": "loaded",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "backend_app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )
