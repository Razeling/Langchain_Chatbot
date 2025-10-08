"""
Web search service for vehicle troubleshooting
Provides fallback search when knowledge base is insufficient
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from backend_app.core.settings import get_settings
from backend_app.models.chat_models import SourceCitation


class WebSearchService:
    """Service for searching automotive troubleshooting information on the web."""

    def __init__(self):
        self.settings = get_settings()
        self.session = None

        # Automotive-focused search terms and domains - UPDATED FOR EUROPEAN MARKETS
        self.automotive_domains = [
            # European Technical Authorities & Official Sources
            "dat.de",  # Deutsche Automobil Treuhand - Official German repair costs
            "dekra.de",  # DEKRA - Major European technical inspection
            "fairgarage.com",  # German workshop price comparison platform
            "autoreparaturen.de",  # German workshop comparison & pricing service
            # European Parts & Service Platforms
            "euro-car-parts.com",  # Major European parts supplier
            "autodoc.com",  # Large European automotive parts retailer
            "motointegrator.com",  # European car parts & service platform
            "gsp.eu",  # European automotive parts & service
            "tecdoc.com",  # European automotive technical database
            # Brand-Specific European Sources
            "mercedesmedic.com",  # Mercedes-Benz specific repair guides
            "benzworld.org",  # Mercedes community (European focus)
            "bimmerworld.com",  # BMW specific (European origin)
            "audizine.com",  # Audi community & technical information
            "ross-tech.com",  # VAG-COM diagnostics (European brands)
            # Regional European Sources
            "autoplenum.de",  # German automotive portal
            "motor-talk.de",  # German automotive community
            "autoscout24.com",  # European car marketplace & info
            "mobile.de",  # German automotive marketplace
            # Keep some global sources for comparison
            "repairpal.com",  # US source (for reference)
            "justanswer.com/car",  # Global expert Q&A
        ]

        self.search_modifiers = [
            # English modifiers
            "car troubleshooting",
            "automotive repair",
            "vehicle diagnostic",
            "car problem solution",
            "auto mechanic advice",
            # European-specific modifiers
            "european car repair",
            "german automotive service",
            "mercedes bmw audi repair",
            "workshop costs europe",
            "automotive service germany",
            # Multi-language modifiers for European markets
            "kfz reparatur",  # German: vehicle repair
            "autoreparatur kosten",  # German: auto repair costs
            "werkstatt preise",  # German: workshop prices
            "vehicle maintenance costs europe",
            "car service pricing germany",
        ]

    async def initialize(self):
        """Initialize the web search service."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
            )
            logger.info("Web search service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize web search service: {e}")
            raise

    async def close(self):
        """Close the web search service."""
        if self.session:
            await self.session.close()

    async def search_automotive_info(
        self, query: str, vehicle_info: Optional[Dict[str, Any]] = None, max_results: int = 3
    ) -> List[SourceCitation]:
        """
        Search for automotive troubleshooting information on the web.

        Args:
            query: Search query
            vehicle_info: Vehicle information to enhance search
            max_results: Maximum number of results to return

        Returns:
            List of source citations from web search
        """
        if not self.session:
            await self.initialize()

        try:
            # Enhance query with automotive context
            enhanced_query = self._enhance_automotive_query(query, vehicle_info)
            logger.info(f"Web search for automotive query: '{enhanced_query}'")

            # Perform multiple search attempts with different strategies
            all_results = []

            # Strategy 1: Direct automotive search
            automotive_results = await self._search_with_modifiers(enhanced_query, max_results)
            all_results.extend(automotive_results)

            # Strategy 2: Brand-specific search if vehicle info available
            if vehicle_info and vehicle_info.get("make"):
                brand_query = f"{vehicle_info['make']} {query}"
                brand_results = await self._search_with_modifiers(brand_query, max_results // 2)
                all_results.extend(brand_results)

            # Eliminate duplicates and limit results
            unique_results = self._deduplicate_results(all_results)
            final_results = unique_results[:max_results]

            logger.info(f"Web search returned {len(final_results)} automotive results")
            return final_results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _enhance_automotive_query(self, query: str, vehicle_info: Optional[Dict[str, Any]]) -> str:
        """Enhance query with automotive and vehicle-specific terms."""
        enhanced = query

        # Add vehicle information if available
        if vehicle_info:
            if vehicle_info.get("make"):
                enhanced = f"{vehicle_info['make']} {enhanced}"
            if vehicle_info.get("model"):
                enhanced = f"{enhanced} {vehicle_info['model']}"
            if vehicle_info.get("year"):
                enhanced = f"{vehicle_info['year']} {enhanced}"

        # Add automotive context
        automotive_terms = ["car", "vehicle", "automotive", "repair", "troubleshooting"]
        query_lower = query.lower()

        if not any(term in query_lower for term in automotive_terms):
            enhanced = f"car {enhanced}"

        return enhanced

    async def _search_with_modifiers(self, query: str, max_results: int) -> List[SourceCitation]:
        """Search with automotive-specific modifiers."""
        results = []

        for modifier in self.search_modifiers[:2]:  # Use first 2 modifiers
            try:
                modified_query = f"{query} {modifier}"
                search_results = await self._perform_web_search(modified_query, max_results)
                results.extend(search_results)

                if len(results) >= max_results:
                    break

            except Exception as e:
                logger.warning(f"Search with modifier '{modifier}' failed: {e}")
                continue

        return results

    async def _perform_web_search(self, query: str, max_results: int) -> List[SourceCitation]:
        """
        Perform actual web search using a search API or web scraping.
        This is a simplified implementation - in production, you'd use:
        - Google Custom Search API
        - Bing Search API
        - SerpAPI
        - Or other search providers
        """
        try:
            # For demo purposes, I'll create mock automotive search results
            # In production, replace this with actual search API calls
            mock_results = await self._get_mock_automotive_results(query, max_results)
            return mock_results

        except Exception as e:
            logger.error(f"Web search API call failed: {e}")
            return []

    async def _get_mock_automotive_results(
        self, query: str, max_results: int
    ) -> List[SourceCitation]:
        """
        Mock automotive search results for demonstration.
        Now provides more specific and relevant content based on the query.
        Replace this with actual search API integration.
        """
        # Simulate API delay
        await asyncio.sleep(0.5)

        query_lower = query.lower()

        # Generate specific mock results based on query content
        mock_data = []

        # Winter preparation queries
        if any(
            word in query_lower
            for word in ["winter", "cold", "snow", "ice", "baltic", "ziemai", "prepare"]
        ):
            mock_data = [
                {
                    "title": "Complete Winter Car Preparation Guide for Baltic Climate",
                    "content": f"For '{query}': Essential winter car preparation includes: 1) Switch to winter tires (mandatory in Baltic countries), 2) Check antifreeze levels and ensure -30°C protection, 3) Install engine block heater for extreme cold, 4) Replace summer wiper fluid with winter formula, 5) Test battery capacity (cold reduces performance by 30%), 6) Check heating system and cabin air filter, 7) Pack emergency winter kit with blankets, food, and phone charger. Baltic winters require specific preparations due to prolonged sub-zero temperatures and coastal humidity.",
                    "url": "https://balticautocare.com/winter-preparation",
                    "domain": "balticautocare.com",
                    "similarity": 0.92,
                },
                {
                    "title": "Battery and Engine Care in Extreme Cold Weather",
                    "content": f"Regarding '{query}': Cold weather drastically affects vehicle performance. Battery capacity drops 30-50% in temperatures below -10°C. Use battery warmer or trickle charger overnight. Engine oil becomes thicker - switch to 0W-30 or 5W-30 winter grade oil. Allow 3-5 minutes warm-up before driving. Check coolant mixture for proper freeze protection. Consider synthetic oil for better cold-weather flow.",
                    "url": "https://coldweatherauto.com/battery-engine-care",
                    "domain": "coldweatherauto.com",
                    "similarity": 0.85,
                },
                {
                    "title": "European Winter Driving Regulations and Safety",
                    "content": f"For '{query}': European regulations require winter tires or chains from October-March in many Baltic countries. Studded tires are permitted but restricted in some cities. Keep emergency supplies: reflective vest, warning triangle, first aid kit. Check local regulations for specific requirements. Winter fuel additives prevent fuel line freezing in diesel vehicles.",
                    "url": "https://europeanwinterdriving.org/regulations",
                    "domain": "europeanwinterdriving.org",
                    "similarity": 0.78,
                },
            ]

        # Mercedes-Benz C55 AMG specific queries
        elif "mercedes" in query_lower and "c55" in query_lower:
            mock_data = [
                {
                    "title": "Mercedes-Benz C55 AMG W203 Common Issues and Solutions",
                    "content": f"For '{query}': Mercedes C55 AMG W203 (2004-2007) specific issues include: 1) Engine mounts deterioration causing vibration, 2) Transmission mounts wear affecting driveability, 3) Suspension bushings degradation, 4) Valve cover gasket leaks, 5) Brake rotor warping due to performance use, 6) Air mass sensor issues, 7) Catalytic converter problems with high-mileage cars. The M113K supercharged engine is generally reliable but requires premium maintenance.",
                    "url": "https://mercedesamgspecialist.com/c55-w203-issues",
                    "domain": "mercedesamgspecialist.com",
                    "similarity": 0.95,
                },
                {
                    "title": "M113K Supercharged Engine Maintenance Guide",
                    "content": f"Regarding '{query}': The M113K 5.4L supercharged V8 in C55 AMG requires: 1) Regular supercharger oil changes every 40,000km, 2) Premium 0W-40 or 5W-40 engine oil, 3) Air filter replacement every 20,000km due to supercharger, 4) Spark plugs every 60,000km with proper heat range, 5) Fuel system cleaning every 40,000km. Monitor supercharger whine - excessive noise indicates wear. Oil consumption normal up to 1L per 2000km under spirited driving.",
                    "url": "https://m113engine.com/supercharged-maintenance",
                    "domain": "m113engine.com",
                    "similarity": 0.91,
                },
                {
                    "title": "C55 AMG Performance and Reliability Analysis",
                    "content": f"For '{query}': C55 AMG reliability analysis shows: Engine (M113K) - very reliable when maintained, transmission (5G-Tronic) - generally robust but service every 60,000km, suspension - AMG components wear faster than standard, brakes - performance pads and rotors last 30-50k km. Common maintenance costs: Engine service €300-500, transmission service €400-600, brake replacement €800-1200. Avoid modified examples.",
                    "url": "https://amganalysis.com/c55-reliability",
                    "domain": "amganalysis.com",
                    "similarity": 0.87,
                },
            ]

        # Mercedes-Benz C63 specific queries
        elif "mercedes" in query_lower and ("c63" in query_lower or "w204" in query_lower):
            mock_data = [
                {
                    "title": "Mercedes-Benz C63 W204 Common Issues and Diagnostics",
                    "content": f"For '{query}': Mercedes C63 W204 (2008-2014) known issues include: 1) Head bolt failure on M156 engine - check for coolant loss and white smoke, 2) Cam adjuster solenoid problems causing rough idle, 3) Transmission mount wear causing vibration, 4) Air suspension compressor failure on AMG models, 5) Brake disc warping due to high-performance use. Use STAR diagnostic tool for accurate fault codes. Regular maintenance intervals critical for AMG engines.",
                    "url": "https://mercedesspecialist.com/c63-w204-issues",
                    "domain": "mercedesspecialist.com",
                    "similarity": 0.94,
                },
                {
                    "title": "M156 Engine Diagnostic and Repair Guide",
                    "content": f"Regarding '{query}': The M156 6.2L V8 in C63 W204 requires specific diagnostic approach: Check engine fault codes first, monitor oil consumption (should be <1L per 1000km), inspect cam adjusters for proper operation, verify fuel system pressure, check intake system for leaks. Common fault codes: P0012/P0022 (cam timing), P0300 (misfire), P2004 (intake manifold). Engine oil MUST be Mercedes 229.5 specification.",
                    "url": "https://m156engine.com/diagnostics",
                    "domain": "m156engine.com",
                    "similarity": 0.89,
                },
                {
                    "title": "European Mercedes-Benz Service and Parts Availability",
                    "content": f"For '{query}': Mercedes-Benz parts availability in Europe is excellent through authorized dealers. Common repair costs: Head bolt replacement €3000-5000, cam adjusters €800-1200, transmission service €400-600. Warranty coverage varies by country. Independent specialists often offer 30-40% savings vs. dealer prices. Use genuine MB parts for critical engine components.",
                    "url": "https://europeanbenzservice.com/parts-costs",
                    "domain": "europeanbenzservice.com",
                    "similarity": 0.82,
                },
            ]

        # General Mercedes-Benz queries (fallback)
        elif "mercedes" in query_lower:
            mock_data = [
                {
                    "title": "Mercedes-Benz General Diagnostic Guide",
                    "content": f"For '{query}': Mercedes-Benz vehicles require specific diagnostic procedures: 1) Use STAR diagnostic system for accurate fault reading, 2) Check service history for proper maintenance intervals, 3) Verify genuine parts usage, 4) Monitor common Mercedes issues like electrical problems, suspension wear, and engine mount deterioration. Refer to model-specific guides for detailed troubleshooting.",
                    "url": "https://mercedesdiagnostics.com/general-guide",
                    "domain": "mercedesdiagnostics.com",
                    "similarity": 0.78,
                }
            ]

        # Engine/starting problems
        elif any(
            word in query_lower
            for word in ["engine", "start", "starting", "ignition", "battery", "cranking"]
        ):
            mock_data = [
                {
                    "title": "Systematic Engine Starting Problem Diagnosis",
                    "content": f"For '{query}': Follow systematic diagnosis: 1) Check battery voltage (12.4V+ at rest, 10V+ while cranking), 2) Listen for fuel pump prime (2-3 seconds after key on), 3) Verify starter engagement and cranking speed, 4) Check for spark at plugs, 5) Scan for fault codes, 6) Test fuel pressure and quality, 7) Inspect air intake system. Modern vehicles have immobilizer systems that can prevent starting - check key programming.",
                    "url": "https://enginediagnostics.com/starting-problems",
                    "domain": "enginediagnostics.com",
                    "similarity": 0.88,
                },
                {
                    "title": "Common Starting System Failures and Solutions",
                    "content": f"Regarding '{query}': Most common starting failures: Weak battery (40% of cases), faulty starter motor (25%), fuel system issues (20%), ignition problems (15%). Test battery load capacity, not just voltage. Check starter current draw (<200A for most engines). Verify fuel pressure meets specification. Inspect ignition coils and spark plugs for wear.",
                    "url": "https://startersystem.org/failures",
                    "domain": "startersystem.org",
                    "similarity": 0.83,
                },
            ]

        # Lithuanian/Baltic specific queries
        elif any(
            word in query_lower
            for word in [
                "prieziura",
                "priežiūra",
                "kaina",
                "kainos",
                "lithuania",
                "baltic",
                "vilnius",
                "kaunas",
            ]
        ):
            mock_data = [
                {
                    "title": "Mercedes-Benz Maintenance Costs in Lithuania and Baltic States",
                    "content": f"For '{query}': Mercedes maintenance costs in Lithuania: C350 annual maintenance €800-1200, including oil service (€150-200), brake service (€300-500), timing chain inspection (€200-300). Parts costs 20-30% higher than Germany due to import duties. Authorized dealers: Vilnius, Kaunas, Klaipėda. Independent workshops 30-40% cheaper. Winter preparation essential - engine block heater, winter tires mandatory November-March.",
                    "url": "https://baltstauto.lt/mercedes-maintenance-costs",
                    "domain": "baltstauto.lt",
                    "similarity": 0.95,
                },
                {
                    "title": "European Car Service Regulations and Pricing in Baltic Region",
                    "content": f"Regarding '{query}': EU regulations require specific service intervals and parts quality. Lithuanian service stations must follow EU safety standards. Average labor rates: €25-45/hour (vs Germany €70-120/hour). Genuine parts availability good through official dealers. Common services: Technical inspection every 2 years, emissions testing, winter tire requirements October-April.",
                    "url": "https://euroservice.lt/car-maintenance-guide",
                    "domain": "euroservice.lt",
                    "similarity": 0.89,
                },
                {
                    "title": "Annual Vehicle Costs Calculator for European Cars",
                    "content": f"For '{query}': Annual vehicle ownership costs in Lithuania for premium cars like Mercedes C350: Insurance €300-600, fuel €1200-1800, maintenance €600-1000, technical inspection €30, road tax €150-300. Total: €2280-3730 per year. Compare to Germany: 20-30% higher due to lower incomes but similar part costs.",
                    "url": "https://autokalkulatorius.lt/annual-costs",
                    "domain": "autokalkulatorius.lt",
                    "similarity": 0.84,
                },
            ]

        # Cost/pricing focused queries (European context)
        elif any(
            word in query_lower
            for word in ["cost", "price", "expensive", "budget", "kaina", "kosten", "prix", "cena"]
        ):
            mock_data = [
                {
                    "title": "European Automotive Repair Cost Database - DAT Pricing",
                    "content": f"For '{query}': Official European repair costs based on DAT database: Labor rates vary by country - Germany €70-120/hr, Eastern Europe €25-50/hr. Parts costs standardized across EU. Mercedes C350 typical repairs: Brake service €300-600, transmission service €400-800, engine service €200-500. Use certified workshops for warranty compliance.",
                    "url": "https://dat.de/reparaturkosten-europa",
                    "domain": "dat.de",
                    "similarity": 0.92,
                },
                {
                    "title": "FairGarage Workshop Price Comparison - European Markets",
                    "content": f"Regarding '{query}': Workshop price comparison across Europe shows significant regional variations. Premium brand services (Mercedes, BMW, Audi) cost 40-60% more than mass market brands. Eastern European workshops offer 30-50% savings vs Western Europe while maintaining EU quality standards. Book online for guaranteed pricing.",
                    "url": "https://fairgarage.com/price-comparison-europe",
                    "domain": "fairgarage.com",
                    "similarity": 0.87,
                },
                {
                    "title": "DEKRA European Service Cost Guidelines",
                    "content": f"For '{query}': DEKRA service cost guidelines for European markets: Factor in regional labor rates, parts availability, and regulatory requirements. Premium vehicles require specialized tools and training - budget 20-30% more than standard vehicles. Always verify workshop certification for warranty work.",
                    "url": "https://dekra.de/european-service-costs",
                    "domain": "dekra.de",
                    "similarity": 0.83,
                },
            ]

        # Default automotive content with European focus
        else:
            mock_data = [
                {
                    "title": "European Automotive Technical Guide - TecDoc Database",
                    "content": f"For issues related to '{query}': European automotive technical database provides comprehensive repair procedures for all EU-market vehicles. Follow manufacturer specifications and EU safety standards. Use TecDoc part numbers for compatibility verification. Consider climate-specific requirements for Northern/Southern Europe.",
                    "url": "https://tecdoc.com/repair-guide-europe",
                    "domain": "tecdoc.com",
                    "similarity": 0.78,
                },
                {
                    "title": "EuroCar Parts - Professional Repair Procedures",
                    "content": f"Regarding '{query}': Professional European repair procedures emphasize safety, environmental compliance, and quality standards. Use OE-spec parts for optimal performance. Consider regional service requirements - winter packages for Nordic countries, emission standards for urban areas. Warranty work requires certified facilities.",
                    "url": "https://euro-car-parts.com/repair-procedures",
                    "domain": "euro-car-parts.com",
                    "similarity": 0.74,
                },
                {
                    "title": "AutoDoc European Maintenance Cost Guide",
                    "content": f"For '{query}': European maintenance costs vary by region but follow similar patterns. Eastern Europe offers cost advantages while maintaining EU quality standards. Premium German brands (Mercedes, BMW, Audi) require specialized service. Budget €800-1500 annually for premium vehicle maintenance in most European markets.",
                    "url": "https://autodoc.com/maintenance-costs-europe",
                    "domain": "autodoc.com",
                    "similarity": 0.71,
                },
            ]

        results = []
        for i, data in enumerate(mock_data[:max_results]):
            citation = SourceCitation(
                title=data["title"],
                content=data["content"],
                similarity=data.get("similarity", 0.80 - (i * 0.05)),
                source=f"Web Search - {data['domain']}",
                metadata={
                    "url": data["url"],
                    "domain": data["domain"],
                    "search_query": query,
                    "retrieved_at": datetime.now().isoformat(),
                    "type": "web_search",
                },
            )
            results.append(citation)

        return results

    def _deduplicate_results(self, results: List[SourceCitation]) -> List[SourceCitation]:
        """Remove duplicate search results based on URL or similar content."""
        seen_urls = set()
        unique_results = []

        for result in results:
            url = result.metadata.get("url", "")
            domain = result.metadata.get("domain", "")

            # Create a unique identifier
            identifier = url or f"{domain}_{result.title}"

            if identifier not in seen_urls:
                seen_urls.add(identifier)
                unique_results.append(result)

        return unique_results

    def should_trigger_web_search(
        self,
        knowledge_base_results: List[SourceCitation],
        min_results: int = 2,
        min_similarity: float = 0.6,
    ) -> bool:
        """
        Determine if web search should be triggered based on knowledge base results.
        Enhanced to detect generic vs specific automotive content.

        Args:
            knowledge_base_results: Results from knowledge base
            min_results: Minimum number of good results needed
            min_similarity: Minimum similarity score for good results

        Returns:
            True if web search should be triggered
        """
        if not knowledge_base_results:
            logger.info("No knowledge base results - triggering web search")
            return True

        # Count high-quality results
        good_results = [r for r in knowledge_base_results if r.similarity >= min_similarity]

        # Specific generic patterns for detection
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
            # Specific generic patterns for detection
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

        for result in good_results:
            title_lower = result.title.lower()
            content_lower = (result.content or "").lower()

            # Check if this is generic automotive content
            is_generic = any(generic_term in title_lower for generic_term in generic_titles)

            # ENHANCED: Additional checks for generic content
            if not is_generic and result.content:
                # Generic maintenance phrases for content analysis
                generic_content_phrases = [
                    "start with visual inspection",
                    "use diagnostic scan tools",
                    "follow established procedures",
                    "repair costs vary significantly",
                    "get multiple quotes",
                    "professional mechanics follow",
                    # Generic maintenance phrases for pattern matching
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

            # ENHANCED: Model-specific validation
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

        # Check if we have learned documents in the results
        learned_results = [
            r
            for r in knowledge_base_results
            if "web learned" in r.title.lower() or "learned" in r.source.lower()
        ]

        # If we have learned documents with good similarity, prefer them
        good_learned_results = [
            r
            for r in learned_results
            if r.similarity >= (min_similarity - 0.1)  # Slightly lower threshold for learned docs
        ]

        if good_learned_results:
            # ENHANCED: Even with learned docs, check if they're actually specific
            specific_learned_results = [r for r in good_learned_results if r not in generic_results]
            if specific_learned_results:
                logger.info(
                    f"Found {len(specific_learned_results)} relevant specific learned documents - skipping web search"
                )
                return False
            else:
                logger.warning(
                    f"Learned documents are generic ({len(good_learned_results)} found) - triggering web search for specificity"
                )
                return True

        # ENHANCED: If query seems model-specific but we only have generic results, trigger web search
        if len(specific_results) == 0 and len(generic_results) > 0:
            logger.warning(
                f"Only generic automotive results found ({len(generic_results)} results), no model-specific content - triggering web search for better specificity"
            )
            return True

        if len(good_results) < min_results:
            logger.info(
                f"Only {len(good_results)} high-quality results (need {min_results}) - triggering web search"
            )
            return True

        # If we have mostly generic results, still trigger web search
        if len(generic_results) > len(specific_results):
            logger.info(
                f"Mostly generic results ({len(generic_results)} generic vs {len(specific_results)} specific) - triggering web search for better specificity"
            )
            return True

        logger.info(
            f"Sufficient specific knowledge base results ({len(specific_results)} specific, {len(generic_results)} generic) - skipping web search"
        )
        return False
