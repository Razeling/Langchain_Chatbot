"""
Diagnostics API routes for the Car Troubleshooting Chatbot
Provides standalone diagnostic operations and additional diagnostic functionality
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from backend_app.models.chat_models import FunctionCall, VehicleInfo
from backend_app.models.vehicle_models import CarDiagnostic, MaintenanceSchedule, RepairEstimate
from backend_app.services.function_calling import FunctionCallingService
from backend_app.services.rag_service import RAGService

router = APIRouter()


async def get_rag_service(request: Request) -> RAGService:
    """Dependency to get RAG service from app state."""
    return request.app.state.rag_service


async def get_function_service() -> FunctionCallingService:
    """Dependency to get function calling service."""
    service = FunctionCallingService()
    await service.initialize()
    return service


@router.post("/diagnose", response_model=CarDiagnostic)
async def diagnose_problem(
    symptoms: str,
    vehicle_info: Optional[VehicleInfo] = None,
    function_service: FunctionCallingService = Depends(get_function_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Standalone diagnostic endpoint for car problems.
    """
    try:
        logger.info(f"Diagnosing problem with symptoms: {symptoms[:100]}...")

        # Get relevant context from RAG
        context = ""
        if rag_service:
            context = await rag_service.get_context_for_query(symptoms)

        # Prepare function call
        func_call = FunctionCall(
            name="diagnoseProblem",
            arguments={
                "symptoms": symptoms,
                "vehicle_info": vehicle_info.dict() if vehicle_info else None,
                "context": context,
            },
        )

        # Execute diagnosis
        result = await function_service.execute_function(func_call)

        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_message)

        return result.result

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")


@router.post("/estimate-cost", response_model=RepairEstimate)
async def estimate_repair_cost(
    repair_description: str,
    vehicle_info: Optional[VehicleInfo] = None,
    location: str = "United States",
    function_service: FunctionCallingService = Depends(get_function_service),
):
    """
    Standalone cost estimation endpoint for repairs.
    """
    try:
        logger.info(f"Estimating cost for: {repair_description[:100]}...")

        # Prepare function call
        func_call = FunctionCall(
            name="estimateRepairCost",
            arguments={
                "repair_description": repair_description,
                "vehicle_info": vehicle_info.dict() if vehicle_info else None,
                "location": location,
            },
        )

        # Execute cost estimation
        result = await function_service.execute_function(func_call)

        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_message)

        return result.result

    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


@router.post("/maintenance-schedule", response_model=MaintenanceSchedule)
async def generate_maintenance_schedule(
    vehicle_info: VehicleInfo,
    current_mileage: int = 0,
    months_ahead: int = 12,
    function_service: FunctionCallingService = Depends(get_function_service),
):
    """
    Standalone maintenance schedule generation endpoint.
    """
    try:
        logger.info(f"Generating maintenance schedule for {vehicle_info.make} {vehicle_info.model}")

        # Prepare function call
        func_call = FunctionCall(
            name="generateMaintenanceSchedule",
            arguments={
                "vehicle_info": vehicle_info.dict(),
                "current_mileage": current_mileage,
                "months_ahead": months_ahead,
            },
        )

        # Execute maintenance schedule generation
        result = await function_service.execute_function(func_call)

        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error_message)

        return result.result

    except Exception as e:
        logger.error(f"Maintenance schedule generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schedule generation failed: {str(e)}")


@router.get("/knowledge-search")
async def search_knowledge_base(
    query: str, limit: int = 5, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Search the car troubleshooting knowledge base.
    """
    try:
        logger.info(f"Searching knowledge base for: {query}")

        # Retrieve documents
        citations = await rag_service.retrieve_documents(query, k=limit)

        # Format results
        results = []
        for citation in citations:
            results.append(
                {
                    "title": citation.title,
                    "content": citation.content,
                    "similarity": citation.similarity,
                    "source": citation.source,
                    "metadata": citation.metadata,
                }
            )

        return {"query": query, "results": results, "total_found": len(results)}

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/knowledge-stats")
async def get_knowledge_base_stats(rag_service: RAGService = Depends(get_rag_service)):
    """
    Get statistics about the knowledge base.
    """
    try:
        stats = rag_service.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@router.post("/bulk-diagnose")
async def bulk_diagnose(
    requests: List[Dict[str, Any]],
    function_service: FunctionCallingService = Depends(get_function_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Perform bulk diagnosis for multiple symptom sets.
    Useful for batch processing or testing.
    """
    try:
        logger.info(f"Processing {len(requests)} bulk diagnosis requests")

        results = []

        for i, request_data in enumerate(requests):
            symptoms = request_data.get("symptoms", "")
            vehicle_info = request_data.get("vehicle_info")

            if not symptoms:
                results.append({"index": i, "error": "No symptoms provided"})
                continue

            try:
                # Get context
                context = await rag_service.get_context_for_query(symptoms)

                # Prepare function call
                func_call = FunctionCall(
                    name="diagnoseProblem",
                    arguments={
                        "symptoms": symptoms,
                        "vehicle_info": vehicle_info,
                        "context": context,
                    },
                )

                # Execute diagnosis
                result = await function_service.execute_function(func_call)

                if result.status == "completed":
                    results.append({"index": i, "diagnosis": result.result.dict()})
                else:
                    results.append({"index": i, "error": result.error_message})

            except Exception as e:
                results.append({"index": i, "error": str(e)})

        return {
            "total_requests": len(requests),
            "results": results,
            "successful": len([r for r in results if "diagnosis" in r]),
            "failed": len([r for r in results if "error" in r]),
        }

    except Exception as e:
        logger.error(f"Bulk diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk diagnosis failed: {str(e)}")


@router.get("/diagnostic-categories")
async def get_diagnostic_categories():
    """
    Get available diagnostic categories and common problems.
    """
    try:
        categories = {
            "Engine": [
                "Won't start",
                "Overheating",
                "Strange noises",
                "Poor performance",
                "Oil leaks",
            ],
            "Brakes": ["Squealing", "Grinding", "Soft pedal", "Vibration", "Pulling to one side"],
            "Transmission": [
                "Hard shifting",
                "Slipping",
                "Leaks",
                "Strange noises",
                "Won't engage",
            ],
            "Electrical": [
                "Dead battery",
                "Charging issues",
                "Lighting problems",
                "Starting issues",
                "Electronic malfunctions",
            ],
            "Tires": [
                "Uneven wear",
                "Low pressure",
                "Vibration",
                "Alignment issues",
                "Balancing problems",
            ],
            "Climate Control": [
                "AC not cooling",
                "No heat",
                "Strange smells",
                "Poor airflow",
                "Temperature control",
            ],
        }

        return {
            "categories": categories,
            "total_categories": len(categories),
            "total_problems": sum(len(problems) for problems in categories.values()),
        }

    except Exception as e:
        logger.error(f"Failed to get diagnostic categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")
