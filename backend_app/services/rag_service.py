"""
RAG (Retrieval-Augmented Generation) service for European car troubleshooting.
Handles document retrieval, similarity search, and context generation.
Specialized for European automotive markets with multi-language support.
"""

import asyncio
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # ChromaDB vector store implementation
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from backend_app.core.settings import get_settings
from backend_app.knowledge_base.european_car_knowledge import (
    KnowledgeDocument,
    get_knowledge_documents,
)
from backend_app.models.chat_models import SourceCitation
from backend_app.services.web_search_service import WebSearchService


class RAGService:
    """RAG service for car troubleshooting knowledge base."""

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        self.web_search_service = None
        self.initialized = False

    async def initialize(self):
        """Initialize the RAG service with embeddings and vector store."""
        try:
            logger.info("Initializing RAG service...")

            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=self.settings.openai_api_key, model=self.settings.openai_embedding_model
            )

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " "],
            )

            # Create vector store directory if it doesn't exist
            os.makedirs(self.settings.vector_store_path, exist_ok=True)

            # Initialize or load vector store
            await self._setup_vector_store()

            # Initialize web search service
            self.web_search_service = WebSearchService()
            await self.web_search_service.initialize()

            self.initialized = True
            logger.info("RAG service with web search initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise

    async def cleanup(self):
        """Cleanup RAG service resources and close database connections."""
        try:
            logger.info("Cleaning up RAG service...")

            # Close vector store connection properly
            if self.vector_store:
                try:
                    # Delete the collection reference to close connection
                    if hasattr(self.vector_store, "_collection"):
                        self.vector_store._collection = None
                    if hasattr(self.vector_store, "_client"):
                        self.vector_store._client = None
                    self.vector_store = None
                    logger.info("Vector store connection closed")
                except Exception as e:
                    logger.warning(f"Error closing vector store: {e}")

            # Cleanup web search service
            if self.web_search_service:
                try:
                    await self.web_search_service.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up web search service: {e}")

            self.initialized = False
            logger.info("RAG service cleanup completed")

        except Exception as e:
            logger.error(f"Error during RAG service cleanup: {e}")

    async def _setup_vector_store(self):
        """Setup vector store with ChromaDB and load existing documents."""
        try:
            settings = get_settings()

            # Use absolute path to ensure it works from any working directory
            vector_store_path = os.path.abspath(settings.vector_store_path)
            vector_store_dir = os.path.join(vector_store_path, "chroma_db")
            logger.info(f"Setting up vector store at absolute path: {vector_store_dir}")

            # Check if vector store directory exists and is valid
            if os.path.exists(vector_store_dir):
                logger.info("Loading existing vector store...")

                # Try to load existing vector store with better error handling
                try:
                    self.vector_store = Chroma(
                        persist_directory=vector_store_dir,
                        embedding_function=self.embeddings,
                        collection_name="automotive_knowledge",
                    )

                    # Test the vector store by trying to access it
                    try:
                        count = self.vector_store._collection.count()
                        logger.info(f"Loaded vector store with {count} documents")

                        if count == 0:
                            logger.warning("Vector store is empty, rebuilding...")
                            await self._create_vector_store()
                        else:
                            logger.info("Vector store loaded successfully")

                    except Exception as e:
                        logger.error(f"Vector store corrupted or inaccessible: {e}")
                        logger.info("Removing corrupted vector store and rebuilding...")

                        # Clean up corrupted vector store
                        try:
                            import shutil

                            shutil.rmtree(vector_store_dir)
                            logger.info("Corrupted vector store removed")
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Could not remove corrupted vector store: {cleanup_error}"
                            )

                        # Recreate vector store
                        await self._create_vector_store()

                except Exception as load_error:
                    logger.error(f"Failed to load vector store: {load_error}")
                    logger.info("Removing corrupted vector store and rebuilding...")

                    # Clean up corrupted vector store
                    try:
                        import shutil

                        if os.path.exists(vector_store_dir):
                            shutil.rmtree(vector_store_dir)
                        logger.info("Corrupted vector store removed")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not remove corrupted vector store: {cleanup_error}")

                    # Recreate vector store
                    await self._create_vector_store()

            else:
                logger.info("Vector store not found, creating new one...")
                await self._create_vector_store()

        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            raise

    async def _create_vector_store(self):
        """Create vector store from comprehensive European car knowledge base documents plus learned documents."""
        try:
            # Get comprehensive European knowledge base documents
            knowledge_docs = get_knowledge_documents()

            # Load previously learned documents from JSONL file
            learned_docs = await self._load_learned_documents()
            if learned_docs:
                knowledge_docs.extend(learned_docs)

            logger.info(
                f"Processing {len(knowledge_docs)} knowledge base + {len(learned_docs)} learned = {len(knowledge_docs)} total knowledge documents..."
            )

            # Convert to LangChain documents
            documents = []
            for doc in knowledge_docs:
                # Convert tags list to comma-separated string for ChromaDB compatibility
                tags_str = ", ".join(doc.tags) if isinstance(doc.tags, list) else str(doc.tags)

                lang_doc = Document(
                    page_content=doc.content,
                    metadata={
                        "id": doc.id,
                        "title": doc.title,
                        "category": doc.category,
                        "tags": tags_str,  # Store as string for ChromaDB compatibility
                        "source": doc.source,
                        "last_updated": doc.last_updated.isoformat(),
                    },
                )
                documents.append(lang_doc)

            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} document chunks")

            # Create vector store
            vector_store_path = os.path.abspath(self.settings.vector_store_path)
            vector_store_dir = os.path.join(vector_store_path, "chroma_db")
            logger.info(f"Creating vector store at: {vector_store_dir}")

            self.vector_store = Chroma.from_documents(
                documents=split_docs, embedding=self.embeddings, persist_directory=vector_store_dir
            )

            # Vector store is automatically persisted with persist_directory
            logger.info("Vector store created and persisted successfully")

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    async def retrieve_documents(
        self, query: str, k: int = None, similarity_threshold: float = None
    ) -> List[SourceCitation]:
        """
        Retrieve relevant documents for a query with vehicle signature filtering and chunk deduplication.

        Args:
            query: Search query
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of source citations with similarity scores, filtered by vehicle signature and deduplicated by document
        """
        if not self.initialized:
            raise RuntimeError("RAG service not initialized")

        k = k or self.settings.retrieval_k
        similarity_threshold = similarity_threshold or self.settings.similarity_threshold

        try:
            logger.info(f"Retrieving documents for query: '{query}' (k={k})")

            # Extract vehicle signature from query for filtering
            query_vehicle_signature = self._extract_vehicle_signature(query)

            # Perform similarity search with more results to allow for filtering and deduplication
            search_k = k * 5  # Get more results to account for chunk deduplication
            results = self.vector_store.similarity_search_with_score(query, k=search_k)

            # Group chunks by document ID and keep the best chunk per document
            document_groups = {}

            for doc, score in results:
                if score <= 1 - similarity_threshold:  # ChromaDB returns distance, not similarity
                    continue

                # Get document ID for grouping chunks
                doc_id = doc.metadata.get("id", "unknown")
                doc_title = doc.metadata.get("title", "Unknown Document")

                # If this is the first chunk for this document, or if this chunk has better similarity
                if (
                    doc_id not in document_groups
                    or (1 - score) > document_groups[doc_id]["similarity"]
                ):
                    document_groups[doc_id] = {
                        "doc": doc,
                        "score": score,
                        "similarity": 1 - score,  # Convert distance to similarity
                        "title": doc_title,
                    }

            # Convert grouped documents back to source citations
            citations = []
            exact_matches = []  # Results that match vehicle signature exactly
            general_matches = []  # Results that don't match vehicle signature but are relevant

            for doc_id, group in document_groups.items():
                doc = group["doc"]
                similarity = group["similarity"]

                citation = SourceCitation(
                    title=doc.metadata.get("title", "Unknown Document"),
                    content=doc.page_content,
                    similarity=similarity,
                    source=doc.metadata.get("source", "Unknown Source"),
                    page=doc.metadata.get("page"),
                    metadata=doc.metadata,
                )

                # Vehicle signature filtering for better matching
                if query_vehicle_signature:
                    content_vehicle_signature = self._extract_vehicle_signature(
                        citation.content + " " + citation.title
                    )

                    if (
                        content_vehicle_signature
                        and query_vehicle_signature == content_vehicle_signature
                    ):
                        exact_matches.append(citation)
                        logger.info(f"🎯 EXACT match: {citation.title[:50]}... (exact vehicle)")
                    elif self._smart_vehicle_signature_match(
                        query_vehicle_signature, content_vehicle_signature
                    ):
                        # Check for brand match at least
                        query_brand = (
                            query_vehicle_signature.split()[0] if query_vehicle_signature else ""
                        )
                        content_brand = (
                            content_vehicle_signature.split()[0]
                            if content_vehicle_signature
                            else ""
                        )

                        if query_brand and content_brand and query_brand == content_brand:
                            general_matches.append(citation)
                            logger.info(f"🏢 BRAND match: {citation.title[:50]}... (same brand)")
                        else:
                            general_matches.append(citation)
                    else:
                        general_matches.append(citation)
                else:
                    general_matches.append(citation)

            # Prioritize exact matches, then general matches
            if exact_matches:
                citations = exact_matches[:k]
                logger.info(
                    f"Vehicle signature filtering: {len(exact_matches)} exact matches, returning top {len(citations)}"
                )
            else:
                citations = general_matches[:k]
                logger.info(
                    f"Vehicle signature filtering: 0 exact matches, {len(general_matches)} general matches, {len(citations)} final results"
                )

            logger.info(f"Retrieved {len(citations)} relevant documents")
            return citations

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []

    async def get_context_for_query(self, query: str, k: int = None) -> str:
        """
        Get context string for a query by retrieving relevant documents.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Formatted context string
        """
        documents = await self.retrieve_documents(query, k)

        if not documents:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for doc in documents:
            context_parts.append(f"Source: {doc.title}\n{doc.content}")

        return "\n\n---\n\n".join(context_parts)

    async def get_combined_context(
        self, query: str, vehicle_info: Optional[Dict[str, Any]] = None, k: int = None
    ) -> Tuple[str, List[SourceCitation]]:
        """
        Get combined context from knowledge base for a query.

        Args:
            query: Search query
            vehicle_info: Vehicle information to enhance query
            k: Number of documents to retrieve

        Returns:
            Tuple of (formatted_context, source_citations)
        """
        # Simple context retrieval from knowledge base only
        sources = await self.retrieve_documents(query, k)

        if not sources:
            return "No relevant information found in the knowledge base.", []

        context_parts = []
        for source in sources:
            context_parts.append(f"Source: {source.title}\n{source.content}")

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    async def retrieve_documents_with_web_fallback(
        self,
        query: str,
        vehicle_info: Optional[Dict[str, Any]] = None,
        k: int = None,
        similarity_threshold: float = None,
        enable_web_search: bool = True,
    ) -> Tuple[List[SourceCitation], List[SourceCitation]]:
        """
        Retrieve documents from knowledge base with web search fallback.

        Args:
            query: Search query
            vehicle_info: Vehicle information for enhanced search
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            enable_web_search: Whether to use web search as fallback

        Returns:
            Tuple of (knowledge_base_results, web_search_results)
        """
        if not self.initialized:
            raise RuntimeError("RAG service not initialized")

        # First, try knowledge base retrieval
        kb_results = await self.retrieve_documents(query, k, similarity_threshold)
        web_results = []

        # Check if web search should be triggered
        if enable_web_search and self.web_search_service and self.settings.enable_web_search:
            should_search_web = self.web_search_service.should_trigger_web_search(
                kb_results,
                min_results=self.settings.web_search_min_kb_results,
                min_similarity=similarity_threshold or self.settings.similarity_threshold,
            )

            if should_search_web:
                logger.info("Knowledge base results insufficient - triggering web search")
                try:
                    web_results = await self.web_search_service.search_automotive_info(
                        query=query,
                        vehicle_info=vehicle_info,
                        max_results=self.settings.web_search_max_results,
                    )
                    logger.info(f"Web search returned {len(web_results)} additional results")

                except Exception as e:
                    logger.error(f"Web search failed: {e}")
                    web_results = []
            else:
                logger.info("Knowledge base results sufficient - skipping web search")

        return kb_results, web_results

    async def learn_from_web_results(
        self, query: str, web_sources: List[SourceCitation], user_feedback: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Learn from web search results by converting them into knowledge base documents.
        Includes deduplication and respects configuration limits for knowledge-first learning.

        Args:
            query: The original user query
            web_sources: Web search results that were useful
            user_feedback: Optional feedback about which results were helpful

        Returns:
            List of knowledge document dicts that were added to the KB
        """
        if not web_sources:
            return []

        # Respect configuration limit for learning
        max_docs = self.settings.max_learned_docs_per_query
        limited_sources = web_sources[:max_docs]

        if len(web_sources) > max_docs:
            logger.info(f"Limiting learning to {max_docs} documents (was {len(web_sources)})")

        added_documents = []

        try:
            logger.info(
                f"Starting knowledge-first learning from {len(limited_sources)} web sources for query: '{query}'"
            )

            # Check for existing similar documents first
            existing_learned_docs = await self._get_existing_learned_documents()

            for i, web_source in enumerate(limited_sources):
                # Check if we already have similar content
                if self._is_similar_content_exists(query, web_source, existing_learned_docs):
                    logger.info(
                        f"Skipping duplicate content for knowledge-first approach: {web_source.title}"
                    )
                    continue

                # VALIDATE WEB CONTENT MATCHES QUERY VEHICLE
                if not self._validate_web_content_for_query(
                    query, web_source.content or "", web_source.title
                ):
                    logger.warning(f"Skipping mismatched vehicle content: {web_source.title}")
                    continue

                # Extract automotive keywords from the query to determine category
                query_lower = query.lower()
                category = "general"
                tags = ["web_learned", "knowledge_first"]

                # Determine category based on query content - enhanced categorization
                if any(
                    word in query_lower
                    for word in ["engine", "start", "ignition", "battery", "motor"]
                ):
                    category = "engine"
                    tags.extend(["engine", "starting", "powertrain"])
                elif any(word in query_lower for word in ["brake", "braking", "stopping", "abs"]):
                    category = "brakes"
                    tags.extend(["brakes", "safety", "stopping"])
                elif any(
                    word in query_lower for word in ["transmission", "gear", "shifting", "clutch"]
                ):
                    category = "transmission"
                    tags.extend(["transmission", "drivetrain", "gearbox"])
                elif any(
                    word in query_lower
                    for word in ["electrical", "light", "fuse", "wire", "battery"]
                ):
                    category = "electrical"
                    tags.extend(["electrical", "lights", "electronics"])
                elif any(
                    word in query_lower
                    for word in ["cooling", "coolant", "radiator", "overheat", "temperature"]
                ):
                    category = "cooling"
                    tags.extend(["cooling", "temperature", "thermostat"])
                elif any(
                    word in query_lower
                    for word in ["winter", "cold", "snow", "ice", "ziemai", "preparation"]
                ):
                    category = "winter_preparation"
                    tags.extend(["winter", "cold_weather", "seasonal"])
                elif any(
                    word in query_lower
                    for word in ["maintenance", "service", "oil", "filter", "inspection"]
                ):
                    category = "maintenance"
                    tags.extend(["maintenance", "service", "routine"])
                elif any(
                    word in query_lower
                    for word in ["diagnostic", "fault", "error", "code", "troubleshoot"]
                ):
                    category = "diagnostics"
                    tags.extend(["diagnostics", "troubleshooting", "fault_codes"])

                # Add brand-specific tags (comprehensive European automotive brands)
                brands_for_tagging = [
                    # German brands
                    "mercedes-benz",
                    "mercedes",
                    "bmw",
                    "audi",
                    "volkswagen",
                    "vw",
                    "porsche",
                    "opel",
                    "smart",
                    # French brands
                    "peugeot",
                    "renault",
                    "citroën",
                    "citroen",
                    "ds",
                    "bugatti",
                    "alpine",
                    # Italian brands
                    "fiat",
                    "ferrari",
                    "lamborghini",
                    "maserati",
                    "alfa romeo",
                    "alfa",
                    "lancia",
                    "pagani",
                    # Swedish brands
                    "volvo",
                    "saab",
                    "scania",
                    "koenigsegg",
                    # British brands
                    "jaguar",
                    "land rover",
                    "range rover",
                    "aston martin",
                    "lotus",
                    "mclaren",
                    # Spanish brands
                    "seat",
                    "cupra",
                    # Czech brands
                    "skoda",
                    "škoda",
                    # Romanian brands
                    "dacia",
                    # Other popular brands in Europe
                    "ford",
                    "toyota",
                    "honda",
                    "nissan",
                    "mazda",
                    "hyundai",
                    "kia",
                    "tesla",
                ]

                for brand in brands_for_tagging:
                    if brand in query_lower:
                        tags.append(brand.replace("-", "_"))  # Use underscore for consistency
                        break

                # Add model-specific tags
                for model in ["c63", "w204", "m156", "e46", "f30", "a4", "golf", "focus"]:
                    if model in query_lower:
                        tags.append(model)

                # Format the web content into a structured knowledge document
                formatted_content = f"""
{web_source.title}

{web_source.content}

Source: {web_source.source}
Retrieved: {web_source.metadata.get('retrieved_at', 'Unknown')}
Knowledge Priority: High (Real-time learned content)

This information was automatically learned from web search to answer the query: "{query}"
This content is prioritized in the knowledge-first learning system for similar future queries.
                """.strip()

                # Create a more deterministic ID based on normalized query
                normalized_query = self._normalize_query_for_dedup(query)
                doc_id = (
                    f"web_learned_{category}_{hash(normalized_query + web_source.title) % 10000}"
                )

                # Create knowledge document dict
                knowledge_doc = {
                    "id": doc_id,
                    "title": f"Web Learned: {web_source.title}",
                    "content": formatted_content,
                    "category": category,
                    "tags": tags,
                    "source": web_source.source,  # web_source.source already contains "Web Search - domain"
                    "original_query": query,
                    "normalized_query": normalized_query,
                    "knowledge_priority": "high",  # Mark as high priority for knowledge-first approach
                    "learned_via": "real_time_web_search",
                    "last_updated": datetime.now().isoformat(),
                }

                # Add to vector store using Document format
                # KnowledgeDocument already imported at top from european_car_knowledge

                # Convert to KnowledgeDocument for vector store
                kb_doc = KnowledgeDocument(
                    id=knowledge_doc["id"],
                    title=knowledge_doc["title"],
                    content=knowledge_doc["content"],
                    category=knowledge_doc["category"],
                    tags=knowledge_doc["tags"],
                    source=knowledge_doc["source"],
                )

                success = await self.add_document(kb_doc)
                if success:
                    added_documents.append(knowledge_doc)
                    existing_learned_docs.append(
                        knowledge_doc
                    )  # Add to local list to prevent duplicates in same batch
                    logger.info(
                        f"✅ Knowledge-first learning: Successfully learned '{knowledge_doc['title']}'"
                    )
                else:
                    logger.warning(
                        f"❌ Knowledge-first learning: Failed to learn '{web_source.title}'"
                    )

            if added_documents:
                logger.info(
                    f"🎯 Knowledge-first learning complete: {len(added_documents)} new documents available for immediate use"
                )

                # Save learned documents to a file for persistence
                await self._save_learned_documents(added_documents)
            else:
                logger.info(
                    "📚 Knowledge-first learning: No new documents needed (all content was duplicates or already exists)"
                )

            return added_documents

        except Exception as e:
            logger.error(f"❌ Knowledge-first learning failed: {e}")
            return []

    def _normalize_query_for_dedup(self, query: str) -> str:
        """
        Comprehensive European multilingual query normalization for cross-language deduplication.
        Handles all major European languages for automotive queries.
        """
        normalized = query.lower().strip()

        # Clean up punctuation for better matching
        normalized = re.sub(r'[?!.,;:"\'()[\]{}]', "", normalized)

        # Filter out common question words in major European languages
        question_words = (
            r"\b("
            r"how|what|when|where|why|should|can|could|would|"
            r"kaip|kas|kada|kur|kodėl|ar|gali|galėtų|"
            r"wie|was|wann|wo|warum|soll|kann|könnte|"
            r"comment|que|quand|où|pourquoi|devrait|peut|"
            r"come|cosa|quando|dove|perché|dovrebbe|può|"
            r"cómo|qué|cuándo|dónde|por qué|debería|puede|"
            r"hoe|wat|wanneer|waar|warum|zou|kan|"
            r"jak|co|kiedy|gdzie|dlaczego|czy|może|"
            r"jak|co|kdy|kde|proč|měl|může|"
            r"hur|vad|när|var|varför|borde|kan|"
            r"hvordan|hva|når|hvor|hvorfor|burde|kan|"
            r"hvordan|hvad|hvornår|hvor|hvorfor|skulle|kan|"
            r"miten|mitä|milloin|missä|miksi|pitäisi|voi|"
            r"hogyan|mit|mikor|hol|miért|kellene|tud|"
            r"cum|ce|când|unde|de ce|ar trebui|poate|"
            r"как|какво|кога|къде|защо|трябва|може|"
            r"kako|što|kada|gdje|zašto|trebao|može|"
            r"πώς|τι|πότε|πού|γιατί|θα πρέπει|μπορεί"
            r")\b"
        )
        normalized = re.sub(question_words, "", normalized)

        # Filter out generic automotive terms in major European languages
        auto_terms = (
            r"\b("
            r"car|vehicle|auto|automobile|"
            r"automobilis|mašina|transportas|"
            r"fahrzeug|wagen|pkw|auto|"
            r"voiture|véhicule|auto|"
            r"macchina|veicolo|auto|"
            r"coche|vehículo|auto|"
            r"auto|voertuig|wagen|"
            r"samochód|pojazd|auto|"
            r"auto|vozidlo|vůz|"
            r"bil|fordon|auto|"
            r"bil|kjøretøy|auto|"
            r"bil|køretøj|auto|"
            r"auto|ajoneuvo|"
            r"autó|jármű|"
            r"mașină|vehicul|auto|"
            r"кола|автомобил|"
            r"auto|vozilo|"
            r"αυτοκίνητο|όχημα"
            r")\b"
        )
        normalized = re.sub(auto_terms, "", normalized)

        # Comprehensive automotive problem terms across major European languages
        problem_terms = {
            # Lithuanian
            "problemos": "problems",
            "gedimas": "failure",
            "klaida": "error",
            "sutrikimas": "malfunction",
            "daznos": "common",
            "tipines": "typical",
            "priežastys": "causes",
            "defektas": "defect",
            "sugadinti": "broken",
            # German
            "probleme": "problems",
            "fehler": "error",
            "störung": "malfunction",
            "häufige": "common",
            "typische": "typical",
            "ursachen": "causes",
            "defekt": "defect",
            "kaputt": "broken",
            "panne": "breakdown",
            # French
            "problèmes": "problems",
            "erreur": "error",
            "panne": "failure",
            "commun": "common",
            "typique": "typical",
            "causes": "causes",
            "défaut": "defect",
            "cassé": "broken",
            "dysfonctionnement": "malfunction",
            # Italian
            "problemi": "problems",
            "errore": "error",
            "guasto": "failure",
            "comune": "common",
            "tipico": "typical",
            "cause": "causes",
            "difetto": "defect",
            "rotto": "broken",
            "malfunzionamento": "malfunction",
            # Spanish
            "problemas": "problems",
            "error": "error",
            "fallo": "failure",
            "común": "common",
            "típico": "typical",
            "causas": "causes",
            "defecto": "defect",
            "roto": "broken",
            "avería": "breakdown",
            # Dutch
            "problemen": "problems",
            "fout": "error",
            "storing": "failure",
            "gewoon": "common",
            "typisch": "typical",
            "oorzaken": "causes",
            "defect": "defect",
            "kapot": "broken",
            "pech": "breakdown",
            # Polish
            "problemy": "problems",
            "błąd": "error",
            "awaria": "failure",
            "wspólny": "common",
            "typowy": "typical",
            "przyczyny": "causes",
            "defekt": "defect",
            "zepsuty": "broken",
            "usterka": "malfunction",
            # Czech
            "problémy": "problems",
            "chyba": "error",
            "porucha": "failure",
            "běžný": "common",
            "typický": "typical",
            "příčiny": "causes",
            "vada": "defect",
            "rozbitý": "broken",
            "závada": "malfunction",
            # Swedish
            "problem": "problems",
            "fel": "error",
            "haveri": "failure",
            "vanlig": "common",
            "typisk": "typical",
            "orsaker": "causes",
            "defekt": "defect",
            "trasig": "broken",
            "störning": "malfunction",
            # Norwegian
            "problemer": "problems",
            "feil": "error",
            "svikt": "failure",
            "vanlig": "common",
            "typisk": "typical",
            "årsaker": "causes",
            "defekt": "defect",
            "ødelagt": "broken",
            "feilfunksjon": "malfunction",
            # Danish
            "problemer": "problems",
            "fejl": "error",
            "svigt": "failure",
            "almindelig": "common",
            "typisk": "typical",
            "årsager": "causes",
            "defekt": "defect",
            "ødelagt": "broken",
            "fejlfunktion": "malfunction",
            # Finnish
            "ongelmat": "problems",
            "virhe": "error",
            "vika": "failure",
            "yleinen": "common",
            "tyypillinen": "typical",
            "syyt": "causes",
            "defekti": "defect",
            "rikki": "broken",
            "häiriö": "malfunction",
            # Hungarian
            "problémák": "problems",
            "hiba": "error",
            "meghibásodás": "failure",
            "közös": "common",
            "tipikus": "typical",
            "okok": "causes",
            "hibás": "defect",
            "törött": "broken",
            "működési": "malfunction",
            # Romanian
            "probleme": "problems",
            "eroare": "error",
            "defecțiune": "failure",
            "comun": "common",
            "tipic": "typical",
            "cauze": "causes",
            "defect": "defect",
            "stricat": "broken",
            "defecțiune": "malfunction",
            # Bulgarian (Cyrillic)
            "проблеми": "problems",
            "грешка": "error",
            "повреда": "failure",
            "общ": "common",
            "типичен": "typical",
            "причини": "causes",
            "дефект": "defect",
            "счупен": "broken",
            "неизправност": "malfunction",
            # Croatian/Serbian
            "problemi": "problems",
            "greška": "error",
            "kvar": "failure",
            "uobičajen": "common",
            "tipičan": "typical",
            "uzroci": "causes",
            "defekt": "defect",
            "pokvarjen": "broken",
            "neispravnost": "malfunction",
            # Greek (basic terms)
            "προβλήματα": "problems",
            "σφάλμα": "error",
            "βλάβη": "failure",
            "κοινός": "common",
            "τυπικός": "typical",
            "αίτια": "causes",
            # Portuguese
            "problemas": "problems",
            "erro": "error",
            "falha": "failure",
            "comum": "common",
            "típico": "typical",
            "causas": "causes",
            "defeito": "defect",
            "quebrado": "broken",
            "avaria": "breakdown",
        }

        # Replace terms with English equivalents for better matching
        for foreign_term, english_term in problem_terms.items():
            normalized = re.sub(rf"\b{re.escape(foreign_term)}\b", english_term, normalized)

        # Clean up extra spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _extract_vehicle_signature(self, query: str) -> str:
        """
        Extract a language-agnostic vehicle signature for cross-language matching.
        Focuses on brand, model, year, and engine codes that are universal across European markets.
        """
        query_lower = query.lower()
        signature_parts = []

        # Extract brand (comprehensive European automotive brands)
        brands = [
            # German brands
            "mercedes-benz",
            "mercedes",
            "bmw",
            "audi",
            "volkswagen",
            "vw",
            "porsche",
            "opel",
            "smart",
            "maybach",
            "mini",
            "rolls-royce",
            "bentley",
            # French brands
            "peugeot",
            "renault",
            "citroën",
            "citroen",
            "ds",
            "bugatti",
            "alpine",
            # Italian brands
            "fiat",
            "ferrari",
            "lamborghini",
            "maserati",
            "alfa romeo",
            "alfa",
            "lancia",
            "pagani",
            "iveco",
            # Swedish brands
            "volvo",
            "saab",
            "scania",
            "koenigsegg",
            # British brands
            "jaguar",
            "land rover",
            "range rover",
            "aston martin",
            "lotus",
            "mclaren",
            "morgan",
            "caterham",
            "tvr",
            # Spanish brands
            "seat",
            "cupra",
            # Czech brands
            "skoda",
            "škoda",
            "tatra",
            # Romanian brands
            "dacia",
            # Dutch brands
            "daf",
            "spyker",
            # Austrian brands
            "ktm",
            # Swiss brands
            "liebherr",
            # Belgian brands
            "minerva",
            # Finnish brands
            "sisu",
            # Also include major non-European but popular in Europe
            "toyota",
            "honda",
            "nissan",
            "mazda",
            "subaru",
            "mitsubishi",
            "suzuki",
            "hyundai",
            "kia",
            "ssangyong",
            "ford",
            "chevrolet",
            "gmc",
            "cadillac",
            "jeep",
            "chrysler",
            "dodge",
            "ram",
            "tesla",
            "lexus",
            "infiniti",
            "acura",
        ]

        for brand in brands:
            if brand in query_lower:
                signature_parts.append(brand.replace("-", "").replace(" ", ""))
                break

        # Extract model codes (comprehensive European market models)
        models = [
            # Mercedes AMG models and performance variants
            "amg gt",
            "gt",
            "gtr",
            "gt r",
            "gt c",
            "gt s",
            "gt 63",
            "gt 43",
            "gt 53",
            "sls",
            "slr",
            "clk",
            "clk63",
            "clk55",
            "clk320",
            "clk350",
            "clk430",
            "clk500",
            # Mercedes regular models
            "c63",
            "c43",
            "c55",
            "c32",
            "c230",
            "c240",
            "c280",
            "c300",
            "c320",
            "c350",
            "c400",
            "c450",
            "e63",
            "e55",
            "e43",
            "e320",
            "e350",
            "e400",
            "e500",
            "e550",
            "s63",
            "s65",
            "s500",
            "s550",
            "s600",
            "s400",
            "s350",
            "s320",
            "cls63",
            "cls55",
            "cls350",
            "cls400",
            "cls500",
            "cls550",
            "sl63",
            "sl65",
            "sl55",
            "sl500",
            "sl550",
            "sl350",
            "sl400",
            "slk55",
            "slk32",
            "slk230",
            "slk280",
            "slk300",
            "slk350",
            "ml63",
            "ml55",
            "ml320",
            "ml350",
            "ml400",
            "ml500",
            "ml550",
            "gl63",
            "gl550",
            "gl450",
            "gl350",
            "gl320",
            "g63",
            "g55",
            "g500",
            "g550",
            "g320",
            "g350",
            "glk350",
            "glk280",
            "glk220",
            "gle63",
            "gle550",
            "gle450",
            "gle350",
            "gle400",
            "gls63",
            "gls550",
            "gls450",
            "gls350",
            "gls400",
            "glc63",
            "glc43",
            "glc300",
            "glc350",
            "glc250",
            "gla45",
            "gla250",
            "gla200",
            "a45",
            "a35",
            "a250",
            "a200",
            "a180",
            "a160",
            "b200",
            "b180",
            "b160",
            # Mercedes chassis codes
            "w204",
            "w205",
            "w211",
            "w212",
            "w213",
            "w220",
            "w221",
            "w222",
            "w219",
            "w203",
            "w202",
            "w201",
            "w124",
            "w123",
            "w210",
            "w140",
            "w126",
            "r230",
            "r231",
            "r129",
            "r107",
            "w463",
            "w164",
            "w166",
            "w167",
            # BMW models
            "m3",
            "m5",
            "m6",
            "m8",
            "m2",
            "m4",
            "x5m",
            "x6m",
            "x3m",
            "x4m",
            "318i",
            "320i",
            "325i",
            "328i",
            "330i",
            "335i",
            "340i",
            "318d",
            "320d",
            "325d",
            "330d",
            "335d",
            "520i",
            "525i",
            "528i",
            "530i",
            "535i",
            "540i",
            "545i",
            "550i",
            "520d",
            "525d",
            "530d",
            "535d",
            "730i",
            "735i",
            "740i",
            "745i",
            "750i",
            "760i",
            "730d",
            "740d",
            "750d",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "z3",
            "z4",
            "z8",
            # BMW chassis codes
            "e46",
            "e90",
            "e91",
            "e92",
            "e93",
            "f30",
            "f31",
            "f32",
            "f33",
            "f34",
            "f36",
            "g20",
            "g21",
            "e39",
            "e60",
            "e61",
            "f10",
            "f11",
            "f07",
            "g30",
            "g31",
            "g32",
            "e38",
            "e65",
            "e66",
            "f01",
            "f02",
            "g11",
            "g12",
            "e36",
            "e30",
            "e21",
            "e28",
            "e34",
            "e32",
            # Audi models
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "a8",
            "q3",
            "q5",
            "q7",
            "q8",
            "tt",
            "r8",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "s8",
            "sq3",
            "sq5",
            "sq7",
            "sq8",
            "rs3",
            "rs4",
            "rs5",
            "rs6",
            "rs7",
            "rs8",
            "rsq3",
            "rsq8",
            # VW models
            "golf",
            "polo",
            "passat",
            "jetta",
            "tiguan",
            "touareg",
            "touran",
            "sharan",
            "beetle",
            "scirocco",
            "eos",
            "phaeton",
            "arteon",
            "atlas",
            "t-roc",
            "troc",
            "gti",
            "gli",
            "r32",
            "r36",
            # Italian supercar/luxury models
            "huracan",
            "huracán",
            "aventador",
            "gallardo",
            "murcielago",
            "murciélago",
            "diablo",
            "countach",
            "urus",
            "revuelto",
            "sterrato",
            "performante",
            "spyder",
            "coupe",
            "458",
            "488",
            "812",
            "f8",
            "sf90",
            "roma",
            "portofino",
            "gtc4lusso",
            "laferrari",
            "california",
            "enzo",
            "testarossa",
            "f40",
            "f50",
            "360",
            "430",
            "550",
            "575",
            "599",
            "quattroporte",
            "ghibli",
            "levante",
            "granturismo",
            "grancabrio",
            "mc20",
            # Other European models
            "focus",
            "fiesta",
            "mondeo",
            "kuga",
            "ecosport",
            "explorer",
            "mustang",
            "206",
            "207",
            "208",
            "306",
            "307",
            "308",
            "406",
            "407",
            "408",
            "508",
            "607",
            "806",
            "807",
            "clio",
            "megane",
            "scenic",
            "laguna",
            "espace",
            "captur",
            "kadjar",
            "500",
            "punto",
            "panda",
            "tipo",
            "bravo",
            "doblo",
            "ducato",
            "corsa",
            "astra",
            "insignia",
            "mokka",
            "crossland",
            "grandland",
            "octavia",
            "fabia",
            "superb",
            "rapid",
            "yeti",
            "kodiaq",
            "karoq",
            "ibiza",
            "leon",
            "toledo",
            "altea",
            "alhambra",
            "ateca",
            "arona",
        ]

        for model in models:
            if model in query_lower:
                signature_parts.append(model)

        # Extract engine codes (comprehensive European engine codes)
        engines = [
            # Mercedes engines
            "m156",
            "m159",
            "m178",
            "m176",
            "m177",
            "m133",
            "m270",
            "m271",
            "m272",
            "m273",
            "m274",
            "m276",
            "m112",
            "m113",
            "m119",
            "m120",
            "m137",
            "m139",
            "m260",
            "m264",
            "m266",
            "m278",
            "m279",
            "om642",
            "om651",
            "om654",
            "om656",
            "om622",
            "om613",
            "om612",
            "om611",
            "om605",
            "om606",
            # BMW engines
            "n52",
            "n54",
            "n55",
            "n20",
            "n26",
            "n13",
            "n14",
            "n16",
            "n18",
            "n46",
            "n62",
            "n63",
            "n73",
            "n74",
            "s52",
            "s54",
            "s62",
            "s63",
            "s65",
            "s85",
            "b38",
            "b46",
            "b48",
            "b58",
            "m52",
            "m54",
            "m56",
            "m57",
            "m67",
            "m70",
            "m73",
            "m88",
            # Audi/VW engines
            "tfsi",
            "tsi",
            "fsi",
            "tdi",
            "sdi",
            "pdi",
            "ea113",
            "ea888",
            "ea827",
            "ea837",
            "1.8t",
            "2.0t",
            "2.5t",
            "3.0t",
            "4.0t",
            "vr6",
            "w8",
            "w12",
            "w16",
            # Other European engines
            "hdi",
            "dci",
            "cdti",
            "dtec",
            "crdi",
            "tdci",
            "jtd",
            "multijet",
            "ecoblue",
            "ecoboost",
            "duratec",
            "zetec",
            "puma",
            "lynx",
            "sigma",
            "fire",
            "multiair",
            "tjet",
        ]

        for engine in engines:
            if engine in query_lower:
                signature_parts.append(engine)

        # Extract year (universal)
        year_match = re.search(r"\b(20[0-2][0-9]|19[8-9][0-9])\b", query)
        if year_match:
            signature_parts.append(year_match.group(1))

        return " ".join(sorted(signature_parts))  # Sort for consistency

    def _extract_vehicle_signature_enhanced(self, query: str) -> tuple:
        """
        Enhanced vehicle signature extraction with model relationships.
        Returns (primary_signature, related_signatures)
        """
        primary_sig = self._extract_vehicle_signature(query)
        related_sigs = []

        query_lower = query.lower()

        # Define model relationships for Mercedes AMG
        if "mercedes" in primary_sig or "mercedesbenz" in primary_sig:
            # C-Class AMG model relationships (same generation compatibility)
            if "c55" in primary_sig:
                related_sigs.extend(["c32 mercedes", "c32 mercedesbenz"])  # W203 generation
            elif "c63" in primary_sig:
                related_sigs.extend(["c55 mercedes", "c55 mercedesbenz"])  # Some shared knowledge
            elif "c32" in primary_sig:
                related_sigs.extend(["c55 mercedes", "c55 mercedesbenz"])

        # E-Class AMG relationships
        if "e55" in primary_sig:
            related_sigs.extend(["e63 mercedes", "e43 mercedes"])
        elif "e63" in primary_sig:
            related_sigs.extend(["e55 mercedes", "e43 mercedes"])

        return primary_sig, related_sigs

    def _smart_vehicle_signature_match(self, query_sig: str, content_sig: str) -> str:
        """
        Smart matching that understands model relationships.
        Returns: 'exact', 'related', 'brand_match', 'different'
        """
        if not query_sig or not content_sig:
            return "unknown"

        if query_sig == content_sig:
            return "exact"

        # Get model relationships
        _, related_sigs = self._extract_vehicle_signature_enhanced(query_sig)

        # Check if content signature matches any related signatures
        if related_sigs:
            for related_sig in related_sigs:
                if related_sig in content_sig or content_sig in related_sig:
                    return "related"

        # Check brand match at least
        query_words = set(query_sig.split())
        content_words = set(content_sig.split())

        # Mercedes brand variations
        mercedes_terms = {"mercedes", "mercedesbenz", "benz"}
        query_has_mercedes = bool(query_words & mercedes_terms)
        content_has_mercedes = bool(content_words & mercedes_terms)

        if query_has_mercedes and content_has_mercedes:
            return "brand_match"

        # BMW brand check
        bmw_terms = {"bmw"}
        query_has_bmw = bool(query_words & bmw_terms)
        content_has_bmw = bool(content_words & bmw_terms)

        if query_has_bmw and content_has_bmw:
            return "brand_match"

        # Audi brand check
        audi_terms = {"audi"}
        query_has_audi = bool(query_words & audi_terms)
        content_has_audi = bool(content_words & audi_terms)

        if query_has_audi and content_has_audi:
            return "brand_match"

        return "different"

    def _validate_web_content_for_query(self, query: str, web_content: str, web_title: str) -> bool:
        """
        Validate that web search content actually matches the query vehicle.
        Prevents learning wrong model information.
        """
        query_sig = self._extract_vehicle_signature(query)
        content_sig = self._extract_vehicle_signature(web_content + " " + web_title)

        if not query_sig:
            return True  # Allow if no specific vehicle in query

        # Check for exact model match first
        query_words = set(query_sig.split())
        content_words = set(content_sig.split())

        # Extract key model identifiers
        query_models = {
            word for word in query_words if word in ["c55", "c63", "c32", "e55", "e63", "m3", "m5"]
        }
        content_models = {
            word
            for word in content_words
            if word in ["c55", "c63", "c32", "e55", "e63", "m3", "m5"]
        }

        # Allow if models match exactly
        if query_models and content_models and query_models == content_models:
            logger.info(
                f"✅ Web content validated: EXACT model match ({query_models}) for {query_sig}"
            )
            return True

        # Reject if different models specified
        if query_models and content_models and query_models != content_models:
            logger.warning(
                f"❌ Web content rejected: Different models - Query: {query_models}, Content: {content_models}"
            )
            return False

        # For general brand content without specific models, use smart matching
        match_type = self._smart_vehicle_signature_match(query_sig, content_sig)

        # Allow exact matches or general brand content when no specific models are specified
        if match_type in ["exact", "brand_match"] and not content_models:
            logger.info(f"✅ Web content validated: {match_type} match for {query_sig}")
            return True
        else:
            logger.warning(
                f"❌ Web content rejected: {match_type} - Query: {query_sig}, Content: {content_sig}"
            )
            return False

    async def _check_cross_language_similarity(
        self, query: str, existing_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Enhanced similarity check that works across languages using semantic similarity.
        """
        # Get vehicle signature for the current query
        current_signature = self._extract_vehicle_signature(query)
        current_normalized = self._normalize_query_for_dedup(query)

        if not current_signature:
            # If no vehicle signature, fall back to normalized query comparison
            return any(
                self._normalize_query_for_dedup(doc.get("original_query", "")) == current_normalized
                for doc in existing_docs
            )

        # Check if any existing document has the same vehicle signature
        for doc in existing_docs:
            existing_query = doc.get("original_query", "")
            existing_signature = self._extract_vehicle_signature(existing_query)
            existing_normalized = doc.get("normalized_query", "")

            if not existing_normalized:
                existing_normalized = self._normalize_query_for_dedup(existing_query)

            # Strong match: Same vehicle signature
            if current_signature and existing_signature and current_signature == existing_signature:
                logger.info(
                    f"🌍 Cross-language match found - Vehicle signature: '{current_signature}'"
                )
                logger.info(f"   Current query: '{query}' -> Existing: '{existing_query}'")
                return True

            # Moderate match: Same normalized terms (after multilingual normalization)
            if (
                current_normalized
                and existing_normalized
                and current_normalized == existing_normalized
            ):
                logger.info(
                    f"🌍 Cross-language match found - Normalized query: '{current_normalized}'"
                )
                return True

        return False

    def _is_similar_content_exists(
        self, query: str, web_source: SourceCitation, existing_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Enhanced similarity check with multilingual support for cross-language deduplication.
        """
        # Use enhanced cross-language similarity check
        if asyncio.iscoroutinefunction(self._check_cross_language_similarity):
            # If we're in an async context, we should use await, but this method isn't async
            # For now, use the synchronous version
            pass

        # Check cross-language similarity
        if self._check_cross_language_similarity_sync(query, existing_docs):
            return True

        # Check content similarity (same title from same source) - this remains the same
        for doc in existing_docs:
            if web_source.title in doc.get("title", "") and web_source.source in doc.get(
                "source", ""
            ):
                logger.info(f"Found duplicate content match: {web_source.title}")
                return True

        return False

    def _check_cross_language_similarity_sync(
        self, query: str, existing_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Synchronous version of cross-language similarity check.
        """
        # Get vehicle signature for the current query
        current_signature = self._extract_vehicle_signature(query)
        current_normalized = self._normalize_query_for_dedup(query)

        if not current_signature:
            # If no vehicle signature, fall back to normalized query comparison
            return any(
                self._normalize_query_for_dedup(doc.get("original_query", "")) == current_normalized
                for doc in existing_docs
            )

        # Check if any existing document has the same vehicle signature
        for doc in existing_docs:
            existing_query = doc.get("original_query", "")
            existing_signature = self._extract_vehicle_signature(existing_query)
            existing_normalized = doc.get("normalized_query", "")

            if not existing_normalized:
                existing_normalized = self._normalize_query_for_dedup(existing_query)

            # Strong match: Same vehicle signature
            if current_signature and existing_signature and current_signature == existing_signature:
                logger.info(
                    f"🌍 Cross-language match found - Vehicle signature: '{current_signature}'"
                )
                logger.info(f"   Current query: '{query}' -> Existing: '{existing_query}'")
                return True

            # Moderate match: Same normalized terms (after multilingual normalization)
            if (
                current_normalized
                and existing_normalized
                and current_normalized == existing_normalized
            ):
                logger.info(
                    f"🌍 Cross-language match found - Normalized query: '{current_normalized}'"
                )
                return True

        return False

    async def _get_existing_learned_documents(self) -> List[Dict[str, Any]]:
        """Get all existing learned documents for deduplication check."""
        try:
            import json

            learned_file = os.path.join(self.settings.vector_store_path, "learned_documents.jsonl")

            if not os.path.exists(learned_file):
                return []

            existing_docs = []
            with open(learned_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            doc_data = json.loads(line)
                            existing_docs.append(doc_data)
                        except json.JSONDecodeError:
                            continue

            return existing_docs

        except Exception as e:
            logger.error(f"Failed to load existing learned documents: {e}")
            return []

    async def _save_learned_documents(self, documents: List[Dict[str, Any]]):
        """Save learned documents to a persistent file for future reference."""
        try:
            import json

            learned_file = os.path.join(self.settings.vector_store_path, "learned_documents.jsonl")
            os.makedirs(os.path.dirname(learned_file), exist_ok=True)

            with open(learned_file, "a", encoding="utf-8") as f:
                for doc in documents:
                    doc_data = {**doc, "learned_at": datetime.now().isoformat()}
                    f.write(json.dumps(doc_data) + "\n")

            logger.info(f"Saved {len(documents)} learned documents to {learned_file}")

        except Exception as e:
            logger.error(f"Failed to save learned documents: {e}")

    async def _load_learned_documents(self) -> List[KnowledgeDocument]:
        """Load previously learned documents from the JSONL file."""
        try:
            import json
            from datetime import datetime

            from backend_app.knowledge_base.european_car_knowledge import KnowledgeDocument

            learned_file = os.path.join(self.settings.vector_store_path, "learned_documents.jsonl")

            if not os.path.exists(learned_file):
                logger.info("No learned documents file found")
                return []

            learned_docs = []
            with open(learned_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        doc_data = json.loads(line)

                        # Convert back to KnowledgeDocument
                        kb_doc = KnowledgeDocument(
                            id=doc_data["id"],
                            title=doc_data["title"],
                            content=doc_data["content"],
                            category=doc_data.get("category", "learned"),
                            tags=doc_data.get("tags", ["web_learned"]),
                            source=doc_data.get("source", "Web Search"),
                            last_updated=datetime.fromisoformat(
                                doc_data.get("last_updated", datetime.now().isoformat())
                            ),
                        )

                        learned_docs.append(kb_doc)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in learned documents: {e}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to convert learned document on line {line_num}: {e}"
                        )

            logger.info(
                f"Successfully loaded {len(learned_docs)} learned documents from {learned_file}"
            )
            return learned_docs

        except Exception as e:
            logger.error(f"Failed to load learned documents: {e}")
            return []

    async def retrieve_documents_with_intelligent_learning(
        self,
        query: str,
        vehicle_info: Optional[Dict[str, Any]] = None,
        k: int = None,
        similarity_threshold: float = None,
        enable_web_search: bool = True,
        auto_learn: bool = True,
    ) -> Tuple[
        List[SourceCitation], List[SourceCitation], List[SourceCitation], List[Dict[str, Any]]
    ]:
        """
        Enhanced document retrieval with intelligent web search and automatic learning.

        Returns 4 separate categories:
        1. internal_knowledge_sources: From european_car_knowledge.py
        2. previously_learned_sources: From learned_documents.jsonl
        3. web_sources: From current web search
        4. newly_learned_documents: What was learned during this query
        """
        if k is None:
            k = self.settings.retrieval_k
        if similarity_threshold is None:
            similarity_threshold = self.settings.similarity_threshold

        logger.info(f"Starting intelligent retrieval for query: '{query}'")

        # Get initial knowledge base results
        kb_results = await self.retrieve_documents(query, k, similarity_threshold)

        # SEPARATE THE SOURCES BY THEIR ORIGIN
        internal_knowledge_sources = []
        previously_learned_sources = []

        for result in kb_results:
            # Check if this is from learned documents (has "Web Search" in source or "learned" indicators)
            is_learned = (
                "Web Search" in (result.source or "")
                or "web learned" in (result.title or "").lower()
                or "learned" in (result.source or "").lower()
            )

            if is_learned:
                previously_learned_sources.append(result)
            else:
                internal_knowledge_sources.append(result)

        logger.info(f"📚 Internal knowledge: {len(internal_knowledge_sources)} sources")
        logger.info(f"🧠 Previously learned: {len(previously_learned_sources)} sources")

        # STEP 2: Evaluate knowledge relevancy with improved logic
        has_relevant_knowledge = self._evaluate_knowledge_relevancy(
            kb_results, query, similarity_threshold
        )

        web_results = []
        learned_documents = []

        if has_relevant_knowledge:
            logger.info(
                f"Found relevant knowledge ({len(kb_results)} results) - using existing knowledge"
            )
            return (
                internal_knowledge_sources,
                previously_learned_sources,
                web_results,
                learned_documents,
            )

        # STEP 3: Knowledge not sufficient - search web if enabled
        if (
            not enable_web_search
            or not self.web_search_service
            or not self.settings.enable_web_search
        ):
            logger.info("Web search disabled - returning available knowledge base results")
            return (
                internal_knowledge_sources,
                previously_learned_sources,
                web_results,
                learned_documents,
            )

        logger.info("Knowledge insufficient - triggering web search for real-time learning")

        try:
            # Perform web search
            web_results = await self.web_search_service.search_automotive_info(
                query=query,
                vehicle_info=vehicle_info,
                max_results=self.settings.web_search_max_results,
            )
            logger.info(f"Web search returned {len(web_results)} results")

            # DEDUPLICATION: Remove web results that are already in knowledge base
            deduplicated_web_results = []
            kb_content_signatures = set()

            # Create signatures for existing knowledge base results
            for kb_result in kb_results:
                # Use title + first 100 chars of content as deduplication key
                kb_signature = f"{kb_result.title}:{kb_result.content[:100]}"
                kb_content_signatures.add(kb_signature)

            # Filter web results to remove duplicates
            for web_result in web_results:
                web_signature = f"{web_result.title}:{web_result.content[:100]}"
                if web_signature not in kb_content_signatures:
                    deduplicated_web_results.append(web_result)
                else:
                    logger.info(f"Skipping duplicate web result: {web_result.title}")

            logger.info(f"After deduplication: {len(deduplicated_web_results)} unique web results")
            web_results = deduplicated_web_results

            # STEP 4: Real-time learning - immediately add web results to knowledge base
            if auto_learn and web_results:
                logger.info("Performing real-time learning from web results")
                learned_documents = await self.learn_from_web_results(query, web_results)
            else:
                learned_documents = []

            if learned_documents:
                logger.info(f"Successfully learned {len(learned_documents)} new documents")

                # STEP 5: Re-search with newly learned knowledge for immediate availability
                logger.info("Re-searching with newly learned knowledge...")
                enhanced_kb_results = await self.retrieve_documents(query, k, similarity_threshold)

                # IMPROVED FIX: Only add learned documents if they weren't found by the vector search
                # This prevents duplicates while ensuring learned content is always available

                # Create a set of titles/content from enhanced search results for deduplication
                found_titles = set()
                found_content_hashes = set()

                for result in enhanced_kb_results:
                    found_titles.add(result.title.lower().strip())
                    # Create a simple content hash for deduplication
                    content_hash = hash(result.content[:200] if result.content else "")
                    found_content_hashes.add(content_hash)

                # Only create learned citations for documents NOT found by vector search
                learned_source_citations = []
                for learned_doc in learned_documents:
                    doc_title = learned_doc.get("title", "").lower().strip()
                    doc_content = learned_doc.get("content", "")
                    doc_content_hash = hash(doc_content[:200] if doc_content else "")

                    # Check if this learned document was already found by vector search
                    title_already_found = doc_title in found_titles
                    content_already_found = doc_content_hash in found_content_hashes

                    if not title_already_found and not content_already_found:
                        # This learned document wasn't found by vector search - add it manually
                        learned_citation = SourceCitation(
                            title=learned_doc.get("title", "Unknown"),
                            content=learned_doc.get("content", ""),
                            source=learned_doc.get("source", "Web Learned"),
                            url=learned_doc.get("url", ""),
                            similarity=0.95,  # High similarity since it was just learned from this query
                            metadata=learned_doc,
                        )
                        learned_source_citations.append(learned_citation)
                        logger.info(
                            f"✅ Added missing learned document: {learned_doc.get('title', 'Unknown')}"
                        )
                    else:
                        logger.info(
                            f"📋 Learned document already found by vector search: {learned_doc.get('title', 'Unknown')}"
                        )

                # Separate enhanced results into internal vs previously learned
                enhanced_internal, enhanced_previously_learned = self._separate_sources_by_origin(
                    enhanced_kb_results
                )

                # Combine with original separated sources
                final_internal_knowledge = self._remove_duplicates(
                    internal_knowledge_sources + enhanced_internal
                )
                final_previously_learned = self._remove_duplicates(
                    previously_learned_sources + enhanced_previously_learned
                )

                # Newly learned content is returned as SourceCitations for immediate display
                newly_learned_sources = []
                for learned_doc in learned_documents:
                    newly_learned_citation = SourceCitation(
                        title=learned_doc.get("title", "Unknown"),
                        content=learned_doc.get("content", ""),
                        source=learned_doc.get("source", "Web Learned"),
                        url=learned_doc.get("url", ""),
                        similarity=0.95,  # High similarity since it was just learned from this query
                        metadata=learned_doc,
                    )
                    newly_learned_sources.append(newly_learned_citation)

                logger.info(f"📚 Final internal knowledge: {len(final_internal_knowledge)} sources")
                logger.info(f"🧠 Final previously learned: {len(final_previously_learned)} sources")
                logger.info(f"🆕 Newly learned: {len(newly_learned_sources)} sources")

                # Return 4 separate categories
                return (
                    final_internal_knowledge,
                    final_previously_learned,
                    newly_learned_sources,
                    learned_documents,
                )
            else:
                logger.info("No new documents learned (duplicates or errors)")

        except Exception as e:
            logger.error(f"Web search or learning failed: {e}")

        # Return original KB results + web results if learning failed
        final_internal_knowledge, final_previously_learned = self._separate_sources_by_origin(
            kb_results
        )
        return final_internal_knowledge, final_previously_learned, web_results, []

    def _evaluate_knowledge_relevancy(
        self, kb_results: List[SourceCitation], query: str, similarity_threshold: float
    ) -> bool:
        """
        Enhanced relevancy evaluation with generic content detection and cross-language support.

        Returns True if existing knowledge is sufficient to answer the query.
        """
        if not kb_results:
            logger.info("No knowledge base results found")
            return False

        # ENHANCED: Use same generic content detection as web search service
        high_quality_results = [r for r in kb_results if r.similarity >= similarity_threshold]

        # Detect generic vs specific content using same logic as web search service
        generic_titles = [
            "automotive diagnostic",
            "repair guide",
            "best practices",
            "cost estimation",
            "professional automotive",
            "general diagnostic",
            "maintenance guide",
            "troubleshooting basics",
            "common problems",
            "automotive service",
            "vehicle inspection",
            # Generic patterns from web search service
            "european brake systems",
            "brake systems and regulations",
            "european automotive",
            "general maintenance",
            "vehicle maintenance",
            "brake maintenance",
            "engine maintenance",
            "automotive regulations",
            "european regulations",
            "safety standards",
            "maintenance standards",
            "automotive compliance",
            "vehicle safety",
            "repair standards",
        ]

        specific_results = []
        generic_results = []

        for result in high_quality_results:
            title_lower = result.title.lower()
            content_lower = (result.content or "").lower()

            # Check if this is generic automotive content
            is_generic = any(generic_term in title_lower for generic_term in generic_titles)

            # Enhanced: Additional checks for generic content
            if not is_generic and result.content:
                # Generic content indicators - expanded list
                generic_content_phrases = [
                    "start with visual inspection",
                    "use diagnostic scan tools",
                    "follow established procedures",
                    "repair costs vary significantly",
                    "get multiple quotes",
                    "professional mechanics follow",
                    # More generic maintenance phrases
                    "check brake pads every",
                    "replace brake fluid every",
                    "follow manufacturer recommendations",
                    "consult service manual",
                    "regular maintenance schedule",
                    "general maintenance guidelines",
                    "standard automotive practice",
                    "typical maintenance intervals",
                    "european safety standards",
                    "regulatory compliance",
                    "standard brake service",
                    "general engine maintenance",
                ]
                generic_phrase_count = sum(
                    1 for phrase in generic_content_phrases if phrase in content_lower
                )
                if generic_phrase_count >= 2:
                    is_generic = True

            # Enhanced: Model-specific validation
            # If the result doesn't mention the specific model/brand in a meaningful way,
            # and the query is model-specific, treat as generic
            if not is_generic:
                # Check if content actually discusses the specific vehicle mentioned in title
                # This catches cases where title suggests specificity but content is generic
                if result.title and any(
                    brand in title_lower for brand in ["mercedes", "bmw", "audi"]
                ):
                    # If title mentions a brand but content doesn't have model-specific details
                    has_specific_model_content = any(
                        model in content_lower
                        for model in [
                            "c55",
                            "c63",
                            "m2",
                            "m3",
                            "m5",
                            "amg",
                            "w203",
                            "w204",
                            "f87",
                            "m113",
                        ]
                    )

                    # If it's supposedly about a specific model but has no model-specific content
                    if not has_specific_model_content and len(content_lower) > 100:
                        logger.warning(
                            f"Source '{result.title}' appears model-specific but lacks specific content - treating as generic"
                        )
                        is_generic = True

            if is_generic:
                generic_results.append(result)
            else:
                specific_results.append(result)

        # Get vehicle signature for cross-language matching
        query_vehicle_signature = self._extract_vehicle_signature(query)

        # Use configurable thresholds
        learned_threshold = self.settings.learned_content_threshold
        high_conf_threshold = self.settings.high_confidence_threshold
        medium_conf_threshold = self.settings.medium_confidence_threshold

        # Count learned document results with lower threshold
        learned_results = [
            r
            for r in kb_results
            if ("web learned" in r.title.lower() or "learned" in r.source.lower())
            and r.similarity >= learned_threshold
        ]

        # Enhanced learned results detection with cross-language vehicle matching
        cross_language_learned_results = []
        if query_vehicle_signature:
            for result in kb_results:
                if "web learned" in result.title.lower() or "learned" in result.source.lower():
                    # Check if this learned result is about the same vehicle
                    result_vehicle_signature = self._extract_vehicle_signature(
                        result.content or result.title
                    )

                    # STRICT MATCHING: Only match if vehicle signatures are EXACTLY the same
                    # This prevents C63 W204 content from being used for AMG GT queries
                    if (
                        result_vehicle_signature
                        and query_vehicle_signature
                        and result_vehicle_signature == query_vehicle_signature
                    ):
                        # Even lower similarity threshold for same vehicle across languages
                        if result.similarity >= (learned_threshold - 0.1):
                            cross_language_learned_results.append(result)
                            logger.info(
                                f"🌍 Cross-language learned content match: {result.title} (similarity: {result.similarity:.3f})"
                            )
                            logger.info(
                                f"   Vehicle signatures match: '{query_vehicle_signature}' == '{result_vehicle_signature}'"
                            )
                    elif (
                        result_vehicle_signature
                        and query_vehicle_signature
                        and result_vehicle_signature != query_vehicle_signature
                    ):
                        # Log when we reject content due to different vehicle signatures
                        logger.info(
                            "❌ Rejecting learned content due to vehicle signature mismatch:"
                        )
                        logger.info(
                            f"   Query: '{query_vehicle_signature}' vs Result: '{result_vehicle_signature}'"
                        )
                        logger.info(f"   Content: {result.title}")
        else:
            # If no vehicle signature in query, be more conservative with learned content
            logger.info(
                f"⚠️  No vehicle signature found in query: '{query}' - limiting learned content matching"
            )

        # High-quality learned results
        high_quality_learned = [r for r in learned_results if r.similarity >= medium_conf_threshold]

        # Combine regular learned + cross-language learned results (avoiding set() for unhashable objects)
        all_learned_results = learned_results[:]
        for result in cross_language_learned_results:
            if not any(
                r.title == result.title and r.source == result.source for r in all_learned_results
            ):
                all_learned_results.append(result)

        all_high_quality_learned = high_quality_learned[:]
        for result in cross_language_learned_results:
            if not any(
                r.title == result.title and r.source == result.source
                for r in all_high_quality_learned
            ):
                all_high_quality_learned.append(result)

        # ENHANCED: Check for generic content issue
        if len(generic_results) > 0 and len(specific_results) == 0:
            logger.warning(
                f"❌ Only generic automotive results found ({len(generic_results)} generic, {len(specific_results)} specific) - knowledge insufficient"
            )
            logger.info(f"   Generic sources: {[r.title for r in generic_results[:3]]}")
            return False

        if len(generic_results) > len(specific_results):
            logger.warning(
                f"❌ Mostly generic results ({len(generic_results)} generic vs {len(specific_results)} specific) - knowledge insufficient for model-specific query"
            )
            return False

        # Evaluate query type for different requirements
        query_lower = query.lower()

        # Classify query types with multilingual support
        is_specific_technical = any(
            word in query_lower
            for word in [
                "mercedes",
                "bmw",
                "audi",
                "volkswagen",
                "c63",
                "w204",
                "m156",
                "error code",
                "fault code",
                "diagnostic",
                "repair cost",
                "engine code",
                "part number",
                "specific model",
                "vin",
                "chassis",
                # Add C350 and other specific models that were missing
                "c350",
                "c300",
                "c250",
                "c200",
                "c180",
                "c160",
                "c43",
                "c32",
                "e350",
                "e300",
                "e250",
                "e200",
                "e400",
                "e500",
                "e550",
                "s350",
                "s400",
                "s500",
                "s550",
                "s600",
                "s320",
                "s280",
                "3.5l",
                "2.0l",
                "1.8l",
                "3.0l",
                "4.0l",
                "5.0l",
                "6.0l",  # Engine sizes
                "petrol engine",
                "diesel engine",
                "turbo",
                "supercharged",
                "v6",
                "v8",
                "v12",  # Engine types
                # Add some common technical terms in other languages
                "klaidos kodas",
                "gedimo kodas",
                "remontas",
                "diagnostika",  # Lithuanian
                "fehlercode",
                "diagnose",
                "reparatur",  # German
            ]
        )

        is_general_automotive = any(
            word in query_lower
            for word in [
                "winter",
                "preparation",
                "maintenance",
                "service",
                "check",
                "problem",
                "oil change",
                "brake",
                "tire",
                "battery",
                "general",
                # Add multilingual equivalents
                "žiema",
                "pasiruošimas",
                "aptarnavimas",
                "tikrinimas",
                "problemos",  # Lithuanian
                "winter",
                "vorbereitung",
                "wartung",
                "service",
                "überprüfung",  # German
            ]
        )

        # PRIORITY 1: High-quality learned content - BUT only if it matches vehicle signature for specific queries
        if self.settings.prioritize_learned_content and all_high_quality_learned:
            # For specific technical queries, require vehicle signature matching
            if is_specific_technical and query_vehicle_signature:
                # Only use learned content if it matches the vehicle signature
                vehicle_matched_learned = [
                    result
                    for result in all_high_quality_learned
                    if self._extract_vehicle_signature(result.content or result.title)
                    == query_vehicle_signature
                ]
                if vehicle_matched_learned:
                    logger.info(
                        f"✅ Vehicle-specific learned content available ({len(vehicle_matched_learned)} results for '{query_vehicle_signature}') - using learned knowledge"
                    )
                    return True
                else:
                    logger.info(
                        f"❌ Found {len(all_high_quality_learned)} learned results, but none match vehicle signature '{query_vehicle_signature}' - triggering web search"
                    )
            else:
                # For general queries, any high-quality learned content is acceptable
                logger.info(
                    f"✅ High-quality learned content available ({len(all_high_quality_learned)} results, {len(cross_language_learned_results)} cross-language) - using learned knowledge"
                )
                return True

        # PRIORITY 2: Cross-language vehicle-specific matches (even with lower similarity)
        if query_vehicle_signature and cross_language_learned_results:
            logger.info(
                f"✅ Cross-language vehicle-specific learned content available ({len(cross_language_learned_results)} results for {query_vehicle_signature})"
            )
            return True

        # PRIORITY 3: Any learned content with reasonable similarity
        if all_learned_results:
            logger.info(
                f"Found {len(all_learned_results)} relevant learned documents ({len(cross_language_learned_results)} cross-language)"
            )

            if is_specific_technical:
                # For technical queries, need at least 1 good learned result
                if (
                    len(all_high_quality_learned) >= 1
                    or len(all_learned_results) >= self.settings.technical_query_min_results
                ):
                    logger.info("✅ Learned content sufficient for specific technical query")
                    return True
            else:
                # For general queries, any learned content with decent similarity is good
                if len(all_learned_results) >= self.settings.general_query_min_results:
                    logger.info("✅ Learned content sufficient for general automotive query")
                    return True

        # PRIORITY 4: High-quality SPECIFIC knowledge base content (not generic)
        if specific_results:
            required_results = (
                self.settings.technical_query_min_results
                if is_specific_technical
                else self.settings.general_query_min_results
            )

            if len(specific_results) >= required_results:
                logger.info(
                    f"✅ High-quality SPECIFIC knowledge base content ({len(specific_results)} specific results) sufficient"
                )
                return True

        # PRIORITY 5: For general queries, medium-quality specific content
        if is_general_automotive and specific_results:
            if len(specific_results) >= 1:  # Just need some specific content for general queries
                logger.info(
                    f"✅ Some specific content ({len(specific_results)} results) sufficient for general query"
                )
                return True

        # Log detailed analysis with generic content info
        logger.info(
            f"❌ Knowledge insufficient - Specific: {len(specific_results)}, "
            f"Generic: {len(generic_results)}, Learned: {len(all_learned_results)} "
            f"(including {len(cross_language_learned_results)} cross-language), "
            f"High Learned: {len(all_high_quality_learned)}, "
            f"Vehicle signature: '{query_vehicle_signature}', "
            f"Query type: {'technical' if is_specific_technical else 'general' if is_general_automotive else 'other'}"
        )
        return False

    def _merge_and_prioritize_results(
        self,
        enhanced_kb_results: List[SourceCitation],
        original_kb_results: List[SourceCitation],
        learned_documents: List[Dict[str, Any]],
    ) -> List[SourceCitation]:
        """
        Merge and prioritize results to put newly learned content first.
        """
        # Get IDs of newly learned documents
        learned_ids = set(doc.get("id", "") for doc in learned_documents)

        # Separate newly learned results from other results
        newly_learned_results = []
        other_results = []

        for result in enhanced_kb_results:
            # Check if this result corresponds to newly learned content
            result_id = result.metadata.get("id", "")
            if result_id in learned_ids:
                newly_learned_results.append(result)
            else:
                other_results.append(result)

        # Add any original results that weren't in the enhanced search
        seen_titles = set(r.title for r in enhanced_kb_results)
        for result in original_kb_results:
            if result.title not in seen_titles:
                other_results.append(result)

        # Sort each group by similarity (highest first)
        newly_learned_results.sort(key=lambda x: x.similarity, reverse=True)
        other_results.sort(key=lambda x: x.similarity, reverse=True)

        # Prioritize: newly learned first, then other results
        final_results = newly_learned_results + other_results

        logger.info(
            f"Merged results: {len(newly_learned_results)} newly learned + {len(other_results)} existing"
        )
        return final_results

    async def add_document(self, document: KnowledgeDocument) -> bool:
        """Add a new document to the vector store."""
        if not self.initialized or not self.vector_store:
            logger.error("RAG service not initialized")
            return False

        try:
            # Convert tags list to string for ChromaDB compatibility
            tags_str = (
                ", ".join(document.tags) if isinstance(document.tags, list) else str(document.tags)
            )

            # Convert to LangChain document
            lang_doc = Document(
                page_content=document.content,
                metadata={
                    "id": document.id,
                    "title": document.title,
                    "category": document.category,
                    "tags": tags_str,
                    "source": document.source,
                    "last_updated": document.last_updated.isoformat(),
                },
            )

            # Split document into chunks
            split_docs = self.text_splitter.split_documents([lang_doc])

            # Add to vector store
            self.vector_store.add_documents(split_docs)

            logger.info(f"Added document '{document.title}' to knowledge base")
            return True

        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG service."""
        if not self.initialized:
            return {"status": "not_initialized"}

        try:
            # Get vector store stats
            collection = self.vector_store._collection
            document_count = collection.count()

            return {
                "status": "initialized",
                "document_count": document_count,
                "embedding_model": self.settings.openai_embedding_model,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "retrieval_k": self.settings.retrieval_k,
                "similarity_threshold": self.settings.similarity_threshold,
            }

        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"status": "error", "error": str(e)}

    async def health_check(self) -> bool:
        """Perform health check on RAG service."""
        try:
            if not self.initialized:
                return False

            # Test retrieval with a simple query
            test_results = await self.retrieve_documents("engine", k=1)
            return len(test_results) > 0

        except Exception as e:
            logger.error(f"RAG health check failed: {e}")
            return False

    def _remove_duplicates(self, results: List[SourceCitation]) -> List[SourceCitation]:
        """Remove duplicate sources while preserving order."""
        seen_titles = set()
        seen_content_hashes = set()
        deduplicated_results = []

        for result in results:
            result_title = result.title.lower().strip()
            result_content_hash = hash(result.content[:200] if result.content else "")

            if result_title not in seen_titles and result_content_hash not in seen_content_hashes:
                seen_titles.add(result_title)
                seen_content_hashes.add(result_content_hash)
                deduplicated_results.append(result)

        return deduplicated_results

    def _separate_sources_by_origin(
        self, results: List[SourceCitation]
    ) -> Tuple[List[SourceCitation], List[SourceCitation]]:
        """Separate sources into internal knowledge vs previously learned."""
        internal_knowledge = []
        previously_learned = []

        for result in results:
            # Check if this is from learned documents
            is_learned = (
                "Web Search" in (result.source or "")
                or "web learned" in (result.title or "").lower()
                or "learned" in (result.source or "").lower()
            )

            if is_learned:
                previously_learned.append(result)
            else:
                internal_knowledge.append(result)

        return internal_knowledge, previously_learned
