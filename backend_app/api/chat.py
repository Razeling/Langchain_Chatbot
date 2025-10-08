"""
Chat API routes for the European Car Troubleshooting Chatbot
Handles chat requests, RAG retrieval, function calling, and streaming responses
Specialized for European markets with focus on Lithuania and Baltic region
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import AsyncOpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend_app.core.settings import get_settings
from backend_app.models.chat_models import (
    ChatRequest,
    ChatResponse,
    FunctionCall,
    SourceCitation,
    StreamChunk,
    VehicleInfo,
)
from backend_app.services.function_calling import FunctionCallingService
from backend_app.services.rag_service import RAGService
from backend_app.services.security_service import ValidationResult, security_service

# Initialize rate limiter
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

# European countries mapping for localization
EUROPEAN_COUNTRIES = {
    "LT": {"name": "Lithuania", "currency": "EUR", "language": "Lithuanian"},
    "LV": {"name": "Latvia", "currency": "EUR", "language": "Latvian"},
    "EE": {"name": "Estonia", "currency": "EUR", "language": "Estonian"},
    "PL": {"name": "Poland", "currency": "PLN", "language": "Polish"},
    "DE": {"name": "Germany", "currency": "EUR", "language": "German"},
    "FR": {"name": "France", "currency": "EUR", "language": "French"},
    "IT": {"name": "Italy", "currency": "EUR", "language": "Italian"},
    "ES": {"name": "Spain", "currency": "EUR", "language": "Spanish"},
    "NL": {"name": "Netherlands", "currency": "EUR", "language": "Dutch"},
    "BE": {"name": "Belgium", "currency": "EUR", "language": "Dutch/French"},
    "AT": {"name": "Austria", "currency": "EUR", "language": "German"},
    "CH": {"name": "Switzerland", "currency": "CHF", "language": "German/French/Italian"},
    "CZ": {"name": "Czech Republic", "currency": "CZK", "language": "Czech"},
    "SK": {"name": "Slovakia", "currency": "EUR", "language": "Slovak"},
    "GB": {"name": "United Kingdom", "currency": "GBP", "language": "English"},
    "NO": {"name": "Norway", "currency": "NOK", "language": "Norwegian"},
    "SE": {"name": "Sweden", "currency": "SEK", "language": "Swedish"},
    "DK": {"name": "Denmark", "currency": "DKK", "language": "Danish"},
    "FI": {"name": "Finland", "currency": "EUR", "language": "Finnish"},
}


async def get_rag_service(request: Request) -> RAGService:
    """Dependency to get RAG service from app state."""
    try:
        if hasattr(request.app.state, "rag_service") and request.app.state.rag_service:
            logger.info("Using RAG service from app state")
            return request.app.state.rag_service
        else:
            logger.error("RAG service not found in app state - this should not happen!")
            logger.error(
                "App state contents: %s",
                dir(request.app.state) if hasattr(request, "app") else "No app",
            )
            raise HTTPException(
                status_code=503, detail="RAG service not properly initialized during startup"
            )
    except Exception as e:
        logger.error(f"Error getting RAG service: {e}")
        raise HTTPException(status_code=503, detail=f"RAG service dependency error: {str(e)}")


async def get_function_service() -> FunctionCallingService:
    """Dependency to get function calling service."""
    service = FunctionCallingService()
    await service.initialize()
    return service


def log_sources(sources, source_type):
    """Helper function to log sources."""
    for i, source in enumerate(sources[:3]):
        logger.info(
            f"  {source_type} Source {i+1}: {source.title} (similarity: {getattr(source, 'similarity', 'N/A'):.3f})"
        )


async def execute_and_log_functions(function_calls, function_service):
    """Execute function calls and log results."""
    executed_functions = []
    for func_call in function_calls:
        executed_func = await function_service.execute_function(func_call)
        executed_functions.append(executed_func)
    return executed_functions


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    function_service: FunctionCallingService = Depends(get_function_service),
):
    """
    Main chat endpoint for European car troubleshooting conversations.
    Handles RAG retrieval, function calling, and response generation.
    """
    try:
        settings = get_settings()
        chat_id = str(uuid.uuid4())

        logger.info(f"Processing European chat request: {chat_id}")

        # SECURITY: Validate input for automotive relevance and security threats
        vehicle_info_dict = (
            chat_request.vehicle_info.__dict__ if chat_request.vehicle_info else None
        )
        validation_result, validation_message = security_service.validate_input(
            chat_request.message, vehicle_info_dict
        )

        if validation_result == ValidationResult.MALICIOUS_INJECTION:
            logger.warning(f"Malicious injection attempt blocked: {chat_request.message[:100]}...")
            return ChatResponse(
                chat_id=chat_id,
                response="⚠️ Security Alert: I can only help with automotive and vehicle-related questions. Please ask about car problems, maintenance, or vehicle information.",
                sources=[],
                function_calls=[],
                vehicle_info=None,
                country=getattr(chat_request, "country", "LT"),
                timestamp=datetime.now(),
            )

        if validation_result == ValidationResult.INVALID_OFF_TOPIC:
            logger.info(f"Off-topic query declined: {chat_request.message[:100]}...")
            return ChatResponse(
                chat_id=chat_id,
                response="🚗 I'm an automotive specialist designed to help with car, vehicle, and automotive-related questions only. I can assist with:\n\n• Car problems and diagnostics\n• Vehicle maintenance and repairs\n• Automotive technical information\n• European car brands and models\n• Parts and components\n\nPlease ask me about something related to cars or vehicles!",
                sources=[],
                function_calls=[],
                vehicle_info=None,
                country=getattr(chat_request, "country", "LT"),
                timestamp=datetime.now(),
            )

        # Use dependency-injected RAG service
        logger.info(f"Using dependency-injected RAG service: {rag_service.initialized}")

        # Extract country and vehicle info
        country_code = getattr(chat_request, "country", "LT")  # Default to Lithuania
        vehicle_info = chat_request.vehicle_info

        # Get country-specific information
        country_info = EUROPEAN_COUNTRIES.get(country_code, EUROPEAN_COUNTRIES["LT"])

        # TRY RAG RETRIEVAL FIRST - Let RAG system handle the query
        logger.info(
            f"Retrieving knowledge with intelligent learning for query: '{chat_request.message}'"
        )

        try:
            # Check RAG service state
            logger.info(
                f"RAG service initialized: {rag_service.initialized if rag_service else False}"
            )
            logger.info(
                f"RAG service vector store: {rag_service.vector_store is not None if rag_service else False}"
            )

            # Use intelligent learning retrieval
            (
                internal_knowledge_sources,
                previously_learned_sources,
                web_sources,
                newly_learned_documents,
            ) = await rag_service.retrieve_documents_with_intelligent_learning(
                query=chat_request.message,
                vehicle_info=vehicle_info.__dict__ if vehicle_info else None,
                k=settings.retrieval_k,
                similarity_threshold=settings.similarity_threshold,
                enable_web_search=settings.enable_web_search,
                auto_learn=True,
            )

            logger.info(f"Retrieved {len(internal_knowledge_sources)} internal knowledge sources")
            logger.info(f"Retrieved {len(previously_learned_sources)} previously learned sources")
            logger.info(f"Retrieved {len(web_sources)} web sources")
            logger.info(f"Learned {len(newly_learned_documents)} new documents from web search")

            # Combine all sources for system prompt (backward compatibility)
            all_knowledge_sources = internal_knowledge_sources + previously_learned_sources

            log_sources(internal_knowledge_sources, "Internal KB")
            log_sources(previously_learned_sources, "Previously Learned")
            log_sources(web_sources, "Web")
            log_sources(newly_learned_documents, "Newly Learned")

        except Exception as e:
            logger.error(f"Error during intelligent knowledge retrieval: {e}")
            internal_knowledge_sources = []
            previously_learned_sources = []
            web_sources = []
            newly_learned_documents = []

        # CHECK IF RAG RESULTS ARE INSUFFICIENT - Only then prompt for vehicle info
        total_sources = (
            len(internal_knowledge_sources)
            + len(previously_learned_sources)
            + len(web_sources)
            + len(newly_learned_documents)
        )

        # SPECIAL CHECK: Even if we have some RAG results, for specific model queries asking about
        # year-sensitive topics (maintenance, costs, problems), we still need the year
        from backend_app.services.vehicle_context_service import VehicleContextService

        vehicle_context_service = VehicleContextService()

        should_prompt, vehicle_prompt = vehicle_context_service.should_prompt_for_vehicle_info(
            chat_request.message, chat_request.vehicle_info
        )

        # Check if this is a year-critical query about a specific model
        message_lower = chat_request.message.lower()
        has_specific_model = any(
            model in message_lower
            for model in [
                "c350",
                "c300",
                "c250",
                "c63",
                "m3",
                "m5",
                "a4",
                "a6",
                "golf",
                "320i",
                "330i",
            ]
        )
        has_year_sensitive_topic = any(
            topic in message_lower
            for topic in [
                "priežiūra",
                "prieziura",
                "maintenance",
                "cost",
                "price",
                "kaina",
                "kainos",
                "problems",
                "issues",
                "problemos",
            ]
        )
        has_year_provided = (chat_request.vehicle_info and chat_request.vehicle_info.year) or any(
            year in chat_request.message
            for year in [
                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
                "2018",
                "2017",
                "2016",
                "2015",
            ]
        )

        if (
            should_prompt
            and has_specific_model
            and has_year_sensitive_topic
            and not has_year_provided
        ):
            logger.info(
                f"Year-critical query detected for specific model, prompting for year: {chat_request.message[:100]}..."
            )
            return ChatResponse(
                chat_id=chat_id,
                response=vehicle_prompt,
                sources=[],
                function_calls=[],
                vehicle_info=chat_request.vehicle_info,
                country=getattr(chat_request, "country", "LT"),
                timestamp=datetime.now(),
            )

        # If no RAG results at all, check for general vehicle prompting
        if total_sources == 0 and should_prompt:
            logger.info(
                f"No RAG results found, prompting for vehicle information: {chat_request.message[:100]}..."
            )
            return ChatResponse(
                chat_id=chat_id,
                response=vehicle_prompt,
                sources=[],
                function_calls=[],
                vehicle_info=chat_request.vehicle_info,
                country=getattr(chat_request, "country", "LT"),
                timestamp=datetime.now(),
            )

        # PROCEED WITH RAG-BASED RESPONSE
        # System message with European car troubleshooting expertise
        system_prompt = build_european_system_prompt(
            vehicle_info, all_knowledge_sources, country_info
        )

        # Determine function calls
        function_calls = []
        executed_functions = []  # Initialize here to avoid scope issues
        if settings.enable_function_calling:
            function_calls = await determine_function_calls(
                message=chat_request.message,
                vehicle_info=vehicle_info,
                function_service=function_service,
                country_code=chat_request.country,
            )

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_request.message},
        ]

        # Execute function calls BEFORE sending to OpenAI so results can be included
        if function_calls and settings.enable_function_calling:
            logger.info(f"Executing {len(function_calls)} function calls before LLM generation")
            executed_functions = await execute_and_log_functions(function_calls, function_service)

            # Add function results to the LLM context
            function_results_text = format_function_results(executed_functions)
            if function_results_text.strip():
                logger.info("Adding function results to LLM context")
                messages.append(
                    {
                        "role": "system",
                        "content": f"FUNCTION EXECUTION RESULTS:\n{function_results_text}\n\nIMPORTANT: Use these function results in your response. Include specific costs, schedules, and recommendations from the executed functions.",
                    }
                )

        # Call OpenAI
        if chat_request.stream:
            return await _handle_streaming_response(
                chat_request,
                messages,
                internal_knowledge_sources,
                previously_learned_sources,
                web_sources,
                newly_learned_documents,
                executed_functions,
                function_service,
            )
        else:
            return await _handle_non_streaming_response(
                chat_request,
                messages,
                internal_knowledge_sources,
                previously_learned_sources,
                web_sources,
                newly_learned_documents,
                executed_functions,
                function_service,
            )

    except Exception as e:
        logger.error(f"European chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def stream_chat_response(
    llm: ChatOpenAI,
    messages: List,
    chat_id: str,
    sources: List,
    function_calls: List,
    vehicle_info: Optional[VehicleInfo],
):
    """Stream chat response with server-sent events."""
    try:
        # Generate streaming response
        response_chunks = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                response_chunks.append(chunk.content)

                # Create stream chunk
                stream_chunk = StreamChunk(id=chat_id, content=chunk.content, finished=False)

                yield f"data: {stream_chunk.json()}\n\n"

        # Send final chunk with complete data
        final_chunk = StreamChunk(
            id=chat_id, content="", finished=True, function_calls=function_calls, sources=sources
        )

        yield f"data: {final_chunk.json()}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = StreamChunk(id=chat_id, content=f"Error: {str(e)}", finished=True)
        yield f"data: {error_chunk.json()}\n\n"


async def extract_vehicle_info(message: str) -> Optional[VehicleInfo]:
    """Extract vehicle information from user message with European brands focus."""
    vehicle_info = VehicleInfo()

    message_lower = message.lower()

    # Extract European car makes
    european_makes = [
        "bmw",
        "mercedes",
        "mercedes-benz",
        "audi",
        "volkswagen",
        "vw",
        "volvo",
        "saab",
        "peugeot",
        "renault",
        "fiat",
        "opel",
        "škoda",
        "skoda",
        "seat",
        "citroën",
        "citroen",
        "alfa romeo",
        "mini",
        "porsche",
        "ferrari",
        "lamborghini",
        "bentley",
        "rolls-royce",
        "jaguar",
        "land rover",
        "range rover",
        "lotus",
        "maserati",
        "lancia",
    ]

    for make in european_makes:
        if make in message_lower:
            if make == "vw":
                vehicle_info.make = "Volkswagen"
            elif make == "mercedes":
                vehicle_info.make = "Mercedes-Benz"
            else:
                vehicle_info.make = make.title()
            break

    # Extract year (look for 4-digit numbers between 1980-2030)
    import re

    year_match = re.search(r"\b(19[8-9]\d|20[0-3]\d)\b", message)
    if year_match:
        vehicle_info.year = int(year_match.group(1))

    # Return None if no vehicle info found
    if not vehicle_info.make and not vehicle_info.year:
        return None

    return vehicle_info


def build_european_system_prompt(
    vehicle_info: Optional[VehicleInfo],
    knowledge_sources: List[SourceCitation],
    country_info: Dict[str, str],
) -> str:
    """Build security-enhanced system prompt for the European car troubleshooting assistant with knowledge-first approach."""
    # Start with security-enhanced base prompt
    base_prompt = security_service.get_security_enhanced_prompt()

    # Add European-specific expertise
    european_context = f"""

EUROPEAN AUTOMOTIVE EXPERTISE:
- Regional Focus: {country_info['name']}
- Currency: {country_info['currency']}
- Language Context: {country_info['language']}

SPECIALIZED EUROPEAN KNOWLEDGE:
- European car brands (BMW, Mercedes, Audi, VW, Volvo, Peugeot, Renault, Fiat, etc.)
- EU emissions standards (Euro 5, Euro 6, Euro 7)
- European technical inspections (MOT, TÜV, CT, etc.)
- Cold climate considerations (especially Nordic/Baltic regions)
- Diesel engine expertise (common in European market)
- European parts availability and OEM specifications
- AdBlue/SCR systems and DPF regeneration

PRICING AND REGULATIONS:
- All costs quoted in {country_info['currency']}
- Include VAT where applicable (typically 20-25% in EU)
- European labor rates and genuine parts pricing
- EU vehicle regulations and type approval
- Country-specific inspection requirements
- Emissions compliance standards
- Right-hand drive considerations for UK/Ireland

KNOWLEDGE-FIRST APPROACH:
- ALWAYS prioritize your automotive knowledge base
- Cite specific sources when providing technical details
- Use previously learned information from similar queries
- Provide comprehensive, detailed automotive answers
- Include specific part numbers, costs, and procedures when available

SAFETY PRIORITY:
- Always prioritize safety in automotive advice
- Recommend professional consultation for complex/dangerous repairs
- Clearly distinguish DIY vs professional-only tasks
- Consider cold European climate in recommendations
"""

    base_prompt += european_context

    if vehicle_info:
        vehicle_text = "\nVEHICLE CONTEXT:\n"
        if vehicle_info.make:
            vehicle_text += f"- Make: {vehicle_info.make}\n"
        if vehicle_info.model:
            vehicle_text += f"- Model: {vehicle_info.model}\n"
        if vehicle_info.year:
            vehicle_text += f"- Year: {vehicle_info.year}\n"
        if vehicle_info.mileage:
            vehicle_text += f"- Mileage: {vehicle_info.mileage} km\n"

        base_prompt += vehicle_text

    if knowledge_sources:
        # Categorize knowledge sources
        learned_sources = [s for s in knowledge_sources if "web learned" in s.title.lower()]
        existing_sources = [s for s in knowledge_sources if "web learned" not in s.title.lower()]

        context_text = "\n=== AVAILABLE KNOWLEDGE ===\n"

        if learned_sources:
            context_text += f"\nPREVIOUSLY LEARNED KNOWLEDGE ({len(learned_sources)} sources):\n"
            context_text += (
                "This information was learned from previous queries and is highly relevant:\n"
            )
            for i, source in enumerate(learned_sources[:3], 1):
                context_text += f"\n{i}. {source.title}\n{source.content[:500]}...\n"

        if existing_sources:
            context_text += f"\nEXISTING KNOWLEDGE BASE ({len(existing_sources)} sources):\n"
            for i, source in enumerate(existing_sources[:2], 1):
                context_text += f"\n{i}. {source.title}\n{source.content[:400]}...\n"

        context_text += "\nCRITICAL INSTRUCTIONS:\n"
        context_text += (
            "- ONLY use the knowledge sources provided above for specific technical details\n"
        )
        context_text += "- DO NOT provide specific engine specifications, horsepower, torque, or part numbers unless they appear in the sources\n"
        context_text += "- If asked about technical specs not in sources, state: 'I don't have specific technical specifications in my current sources'\n"
        context_text += (
            "- Cite specific sources when providing any technical details or specifications\n"
        )
        context_text += "- For general automotive advice, you may use general knowledge, but always prioritize provided sources\n"
        context_text += "- If using learned knowledge, mention it was from previous research\n"
        context_text += "- When sources are insufficient for technical details, recommend consulting official documentation\n"

        base_prompt += context_text

    return base_prompt


async def determine_function_calls(
    message: str,
    vehicle_info: Optional[VehicleInfo],
    function_service: FunctionCallingService,
    country_code: str = "LT",
) -> List[FunctionCall]:
    """Determine which functions to call based on user message and European context."""
    function_calls = []
    message_lower = message.lower()

    # Check for diagnostic keywords
    diagnostic_keywords = [
        "problem",
        "issue",
        "symptom",
        "trouble",
        "noise",
        "won't start",
        "grinding",
        "squealing",
        "dpf",
        "glow plug",
        "adblue",
    ]
    if any(keyword in message_lower for keyword in diagnostic_keywords):
        function_calls.append(
            FunctionCall(
                name="diagnoseProblem",
                arguments={
                    "symptoms": message,
                    "vehicle_info": vehicle_info.dict() if vehicle_info else None,
                },
            )
        )

    # Check for cost estimation keywords
    cost_keywords = [
        "cost",
        "price",
        "estimate",
        "how much",
        "repair cost",
        "expensive",
        "euro",
        "eur",
    ]
    if any(keyword in message_lower for keyword in cost_keywords):
        function_calls.append(
            FunctionCall(
                name="estimateRepairCost",
                arguments={
                    "repair_description": message,
                    "vehicle_info": vehicle_info.dict() if vehicle_info else None,
                    "location": EUROPEAN_COUNTRIES.get(country_code, {}).get("name", "Lithuania"),
                },
            )
        )

    # Check for maintenance keywords
    maintenance_keywords = [
        "maintenance",
        "service",
        "schedule",
        "when to",
        "oil change",
        "tune up",
        "technical inspection",
        "mot",
        "tüv",
    ]
    if any(keyword in message_lower for keyword in maintenance_keywords):
        if vehicle_info:
            function_calls.append(
                FunctionCall(
                    name="generateMaintenanceSchedule",
                    arguments={
                        "vehicle_info": vehicle_info.dict(),
                        "current_mileage": vehicle_info.mileage or 0,
                        "country": country_code,
                    },
                )
            )

    # Check for European regulations keywords
    regulation_keywords = [
        "regulations",
        "legal",
        "law",
        "compliance",
        "emission",
        "euro 6",
        "euro 7",
        "inspection",
        "mot",
        "tüv",
        "contrôle technique",
        "emission zone",
        "lez",
        "ulez",
        "adblue",
        "dpf",
        "scr",
        "low emission",
        "city center",
        "ban",
        "restriction",
        "eu standard",
        "european standard",
        "legal requirement",
    ]
    if any(keyword in message_lower for keyword in regulation_keywords):
        regulation_type = "general"
        if any(word in message_lower for word in ["emission", "euro", "adblue", "dpf"]):
            regulation_type = "emissions"
        elif any(word in message_lower for word in ["inspection", "mot", "tüv", "contrôle"]):
            regulation_type = "inspection"

        function_calls.append(
            FunctionCall(
                name="getEuropeanRegulations",
                arguments={
                    "regulation_type": regulation_type,
                    "country": EUROPEAN_COUNTRIES.get(country_code, {}).get("name", "Lithuania"),
                    "vehicle_info": vehicle_info.dict() if vehicle_info else None,
                },
            )
        )

    return function_calls


def format_function_results(function_calls: List[FunctionCall]) -> str:
    """Format function call results for inclusion in LLM prompt."""
    results_text = ""

    for func_call in function_calls:
        if func_call.status == "completed" and func_call.result:
            results_text += f"\n{func_call.name} Result:\n"

            if hasattr(func_call.result, "dict"):
                # Pydantic model - convert to dict
                result_dict = func_call.result.dict()
                results_text += json.dumps(result_dict, indent=2, default=str)
            else:
                results_text += str(func_call.result)

            results_text += "\n"

    return results_text


@router.get("/chat/health")
async def chat_health_check(rag_service: RAGService = Depends(get_rag_service)):
    """Health check endpoint for European chat service."""
    try:
        # Check RAG service health
        rag_healthy = await rag_service.health_check()

        return {
            "status": "healthy" if rag_healthy else "degraded",
            "rag_service": "healthy" if rag_healthy else "unhealthy",
            "region": "Europe",
            "primary_market": "Lithuania",
            "supported_countries": len(EUROPEAN_COUNTRIES),
            "timestamp": str(asyncio.get_event_loop().time()),
        }

    except Exception as e:
        logger.error(f"European chat health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@router.get("/chat/stats")
async def chat_stats(rag_service: RAGService = Depends(get_rag_service)):
    """Get European chat service statistics."""
    try:
        rag_stats = rag_service.get_stats()

        return {
            "rag_stats": rag_stats,
            "available_functions": [
                "diagnoseProblem",
                "estimateRepairCost",
                "generateMaintenanceSchedule",
                "getEuropeanRegulations",
            ],
            "supported_countries": list(EUROPEAN_COUNTRIES.keys()),
            "primary_currencies": ["EUR", "PLN", "CZK", "GBP", "NOK", "SEK", "DKK", "CHF"],
            "specializations": [
                "European brands",
                "Diesel engines",
                "Cold climate",
                "EU regulations",
            ],
            "web_search_enabled": settings.enable_web_search,
        }

    except Exception as e:
        logger.error(f"Failed to get European chat stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


async def _handle_non_streaming_response(
    chat_request: ChatRequest,
    messages: List[Dict[str, str]],
    internal_knowledge_sources: List[SourceCitation],
    previously_learned_sources: List[SourceCitation],
    web_sources: List[SourceCitation],
    newly_learned_documents: List[Dict[str, Any]],
    executed_functions: List[FunctionCall],
    function_service: FunctionCallingService,
) -> ChatResponse:
    """Handle non-streaming chat response with properly categorized sources."""

    try:
        # Create OpenAI client
        client = AsyncOpenAI(api_key=get_settings().openai_api_key)

        # Generate response
        response = await client.chat.completions.create(
            model=get_settings().openai_model,
            messages=messages,
            temperature=get_settings().openai_temperature,
            max_tokens=2500,
        )

        content = response.choices[0].message.content

        # Convert newly learned documents to proper format
        learned_docs_dict = []
        for doc in newly_learned_documents:
            learned_docs_dict.append(
                {
                    "id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "category": doc.get("category", "Unknown"),
                    "tags": doc.get("tags", []),
                    "source": doc.get("source", ""),
                    "learned_at": doc.get("learned_at", datetime.now().isoformat()),
                }
            )

        # Convert newly learned documents to SourceCitations for display
        newly_learned_sources = []
        for doc in newly_learned_documents:
            newly_learned_sources.append(
                SourceCitation(
                    title=doc.get("title", "Unknown"),
                    content=doc.get("content", ""),
                    source=doc.get("source", "Web Learned"),
                    url=doc.get("url", ""),
                    similarity=0.95,
                    metadata=doc,
                )
            )

        # Combine all sources for legacy compatibility
        all_sources = (
            internal_knowledge_sources
            + previously_learned_sources
            + web_sources
            + newly_learned_sources
        )

        return ChatResponse(
            chat_id=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            response=content,
            sources=all_sources,  # Legacy field - all sources combined
            # Clean source categorization system - each source appears in exactly ONE category
            pure_knowledge_sources=internal_knowledge_sources,  # 📚 Internal Knowledge (european_car_knowledge.py)
            web_learned_sources=previously_learned_sources,  # 🧠 Previously Learned (learned_documents.jsonl)
            web_sources=newly_learned_sources,  # 🆕 Newly Learned (current query)
            # Legacy fields for backward compatibility
            knowledge_sources=internal_knowledge_sources + previously_learned_sources,
            learned_documents=learned_docs_dict,
            function_calls=executed_functions,
            vehicle_info=chat_request.vehicle_info,
            country=getattr(chat_request, "country", "LT"),
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error in non-streaming response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


async def _handle_streaming_response(
    chat_request: ChatRequest,
    messages: List[Dict[str, str]],
    internal_knowledge_sources: List[SourceCitation],
    previously_learned_sources: List[SourceCitation],
    web_sources: List[SourceCitation],
    newly_learned_documents: List[Dict[str, Any]],
    executed_functions: List[FunctionCall],
    function_service: FunctionCallingService,
):
    """Handle streaming chat response with learned documents."""

    async def generate():
        try:
            # Convert newly learned documents to SourceCitations for display
            newly_learned_sources = []
            for doc in newly_learned_documents:
                newly_learned_sources.append(
                    SourceCitation(
                        title=doc.get("title", "Unknown"),
                        content=doc.get("content", ""),
                        source=doc.get("source", "Web Learned"),
                        url=doc.get("url", ""),
                        similarity=0.95,
                        metadata=doc,
                    )
                )

            # Combine all sources for legacy compatibility
            all_knowledge_sources = internal_knowledge_sources + previously_learned_sources

            # Send initial metadata with properly categorized sources
            initial_data = {
                "type": "metadata",
                "data": {
                    # Clean source categorization system - each source appears in exactly ONE category
                    "pure_knowledge_sources": [
                        source.dict() for source in internal_knowledge_sources
                    ],  # 📚 Internal Knowledge
                    "web_learned_sources": [
                        source.dict() for source in previously_learned_sources
                    ],  # 🧠 Previously Learned
                    "web_sources": [
                        source.dict() for source in newly_learned_sources
                    ],  # 🆕 Newly Learned
                    # Legacy fields for backward compatibility
                    "knowledge_sources": [source.dict() for source in all_knowledge_sources],
                    "learned_documents": [
                        {
                            "id": doc.get("id", ""),
                            "title": doc.get("title", ""),
                            "category": doc.get("category", "Unknown"),
                            "tags": doc.get("tags", []),
                            "source": doc.get("source", ""),
                            "learned_at": doc.get("learned_at", datetime.now().isoformat()),
                        }
                        for doc in newly_learned_documents
                    ],
                    "function_calls": [call.dict() for call in executed_functions],
                    "learning_enabled": len(newly_learned_documents) > 0,
                },
            }
            yield f"data: {json.dumps(initial_data)}\n\n"

            # Create OpenAI client
            client = AsyncOpenAI(api_key=get_settings().openai_api_key)

            # Stream the response
            stream = await client.chat.completions.create(
                model=get_settings().openai_model,
                messages=messages,
                temperature=get_settings().openai_temperature,
                max_tokens=2500,  # Increased for detailed automotive responses
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_data = {
                        "type": "content",
                        "data": {"content": chunk.choices[0].delta.content},
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_data = {"type": "error", "data": {"error": str(e)}}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Removed problematic export endpoints that were causing issues
