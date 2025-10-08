#!/usr/bin/env python3
"""
Startup script for the Car Troubleshooting Chatbot Backend.
Run this from the project root directory.
"""

import os
import sys
from pathlib import Path

# FORCE SET SIMILARITY THRESHOLD BEFORE IMPORTING ANYTHING
os.environ["SIMILARITY_THRESHOLD"] = "0.6"

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main application
if __name__ == "__main__":
    import uvicorn

    from backend_app.core.settings import get_settings

    settings = get_settings()

    print("🚗 Starting Car Troubleshooting Chatbot Backend...")
    print(f"📍 Server: http://{settings.host}:{settings.port}")
    print(f"📚 API Docs: http://{settings.host}:{settings.port}/docs")
    print(f"🔧 Environment: {settings.environment}")

    uvicorn.run(
        "backend_app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )
