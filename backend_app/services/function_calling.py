"""
Function calling service for car troubleshooting
Implements specialized diagnostic functions for automotive issues
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from backend_app.core.settings import get_settings
from backend_app.models.chat_models import FunctionCall
from backend_app.models.vehicle_models import (
    CarDiagnostic,
    MaintenanceItem,
    MaintenanceSchedule,
    RepairEstimate,
)


class FunctionCallingService:
    """Service for executing car diagnostic functions."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = None
        self.available_functions = {
            "diagnoseProblem": self.diagnose_problem,
            "estimateRepairCost": self.estimate_repair_cost,
            "generateMaintenanceSchedule": self.generate_maintenance_schedule,
            "getEuropeanRegulations": self.get_european_regulations,
        }

    async def initialize(self):
        """Initialize the function calling service."""
        try:
            self.llm = ChatOpenAI(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                temperature=self.settings.openai_temperature,
            )
            logger.info("Function calling service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize function calling service: {e}")
            raise

    async def execute_function(self, function_call: FunctionCall) -> FunctionCall:
        """Execute a function call and return the result."""
        start_time = asyncio.get_event_loop().time()

        try:
            if function_call.name not in self.available_functions:
                raise ValueError(f"Unknown function: {function_call.name}")

            logger.info(f"Executing function: {function_call.name}")

            # Execute the function
            func = self.available_functions[function_call.name]
            result = await func(**function_call.arguments)

            # Update function call with result
            function_call.result = result
            function_call.status = "completed"
            function_call.execution_time = asyncio.get_event_loop().time() - start_time

            logger.info(f"Function {function_call.name} completed")

        except Exception as e:
            function_call.status = "error"
            function_call.error_message = str(e)
            function_call.execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Function {function_call.name} failed: {e}")

        return function_call

    async def diagnose_problem(
        self,
        symptoms: str,
        vehicle_info: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> CarDiagnostic:
        """Diagnose car problems based on symptoms."""
        try:
            # Create basic diagnosis for now
            diagnosis = CarDiagnostic(
                problem="Engine Problem Detected",
                severity="medium",
                description=f"Based on symptoms: {symptoms}",
                possible_causes=["Battery issue", "Fuel system", "Ignition problem"],
                recommended_actions=["Check battery", "Inspect fuel system", "Test ignition"],
                urgency="Schedule inspection within a week",
                safety_notes=["Do not ignore unusual sounds"],
            )

            return diagnosis

        except Exception as e:
            logger.error(f"Failed to diagnose problem: {e}")
            return CarDiagnostic(
                problem="Diagnostic Error",
                severity="medium",
                description=f"Error during diagnosis: {str(e)}",
                possible_causes=["System error"],
                recommended_actions=["Try again or consult professional"],
                urgency="Seek professional diagnosis",
                safety_notes=["Have vehicle inspected"],
            )

    async def estimate_repair_cost(
        self,
        repair_description: str,
        vehicle_info: Optional[Dict[str, Any]] = None,
        location: str = "Lithuania",
    ) -> RepairEstimate:
        """Estimate repair costs for European markets with real pricing data."""
        try:
            # European labor rates by country (EUR per hour)
            labor_rates = {
                "Lithuania": 45,
                "Latvia": 42,
                "Estonia": 48,
                "Poland": 35,
                "Czech Republic": 40,
                "Slovakia": 38,
                "Hungary": 42,
                "Slovenia": 55,
                "Croatia": 35,
                "Germany": 110,
                "Austria": 95,
                "Switzerland": 140,
                "France": 85,
                "Belgium": 75,
                "Netherlands": 90,
                "Italy": 70,
                "Spain": 65,
                "Portugal": 55,
                "Sweden": 120,
                "Norway": 130,
                "Denmark": 115,
                "Finland": 105,
                "UK": 85,
                "Ireland": 80,
            }

            # VAT rates by country
            vat_rates = {
                "Lithuania": 0.21,
                "Latvia": 0.21,
                "Estonia": 0.20,
                "Poland": 0.23,
                "Czech Republic": 0.21,
                "Slovakia": 0.20,
                "Hungary": 0.27,
                "Slovenia": 0.22,
                "Croatia": 0.25,
                "Germany": 0.19,
                "Austria": 0.20,
                "Switzerland": 0.077,
                "France": 0.20,
                "Belgium": 0.21,
                "Netherlands": 0.21,
                "Italy": 0.22,
                "Spain": 0.21,
                "Portugal": 0.23,
                "Sweden": 0.25,
                "Norway": 0.25,
                "Denmark": 0.25,
                "Finland": 0.24,
                "UK": 0.20,
                "Ireland": 0.23,
            }

            base_labor_rate = labor_rates.get(location, 50)  # Default to 50 EUR/hr
            vat_rate = vat_rates.get(location, 0.21)  # Default to 21% VAT

            # Analyze repair type and estimate parts/labor
            repair_lower = repair_description.lower()

            # Common European repair estimates (base prices before VAT)
            if any(word in repair_lower for word in ["brake", "pad", "disc", "rotor"]):
                if "brake pads" in repair_lower:
                    part_cost = (
                        120
                        if vehicle_info
                        and vehicle_info.get("make", "").lower() in ["bmw", "mercedes", "audi"]
                        else 80
                    )
                    labor_hours = 1.5
                    difficulty = "moderate"
                    tools_required = ["Socket set", "Brake piston tool", "Jack and stands"]
                    estimated_time = "1.5-2 hours"
                elif "brake disc" in repair_lower or "rotor" in repair_lower:
                    part_cost = (
                        200
                        if vehicle_info
                        and vehicle_info.get("make", "").lower() in ["bmw", "mercedes", "audi"]
                        else 150
                    )
                    labor_hours = 2.5
                    difficulty = "moderate"
                    tools_required = [
                        "Socket set",
                        "Brake piston tool",
                        "Disc removal tools",
                        "Jack and stands",
                    ]
                    estimated_time = "2.5-3 hours"
                else:
                    part_cost = 100
                    labor_hours = 2.0
                    difficulty = "moderate"
                    tools_required = ["Basic brake tools"]
                    estimated_time = "2-3 hours"

            elif any(word in repair_lower for word in ["oil", "filter", "service"]):
                premium_brand = vehicle_info and vehicle_info.get("make", "").lower() in [
                    "bmw",
                    "mercedes",
                    "audi",
                    "volvo",
                ]
                part_cost = 80 if premium_brand else 50
                labor_hours = 0.75
                difficulty = "easy"
                tools_required = ["Oil drain pan", "Socket wrench", "Oil filter wrench"]
                estimated_time = "30-45 minutes"

            elif any(word in repair_lower for word in ["timing", "belt", "chain"]):
                interference_engine = vehicle_info and vehicle_info.get("make", "").lower() in [
                    "audi",
                    "volkswagen",
                ]
                part_cost = 400 if interference_engine else 250
                labor_hours = 8.0 if interference_engine else 6.0
                difficulty = "expert"
                tools_required = ["Timing tools", "Engine support", "Specialized timing equipment"]
                estimated_time = "6-10 hours"

            elif any(word in repair_lower for word in ["clutch"]):
                manual_transmission = True  # Assume manual for clutch repair
                part_cost = (
                    600
                    if vehicle_info and vehicle_info.get("make", "").lower() in ["bmw", "mercedes"]
                    else 400
                )
                labor_hours = 8.0
                difficulty = "expert"
                tools_required = ["Transmission jack", "Clutch alignment tool", "Engine support"]
                estimated_time = "8-12 hours"

            elif any(word in repair_lower for word in ["suspension", "shock", "strut"]):
                part_cost = (
                    350
                    if vehicle_info
                    and vehicle_info.get("make", "").lower() in ["bmw", "mercedes", "audi"]
                    else 200
                )
                labor_hours = 4.0
                difficulty = "moderate"
                tools_required = ["Spring compressor", "Strut mount tools", "Jack and stands"]
                estimated_time = "3-5 hours"

            elif any(word in repair_lower for word in ["dpf", "particulate", "filter"]):
                # Common in European diesels
                part_cost = 1200
                labor_hours = 3.0
                difficulty = "moderate"
                tools_required = ["DPF removal tools", "Diagnostic scanner"]
                estimated_time = "3-4 hours"

            elif any(word in repair_lower for word in ["adblue", "scr", "def"]):
                # European diesel emission system
                part_cost = 800
                labor_hours = 4.0
                difficulty = "moderate"
                tools_required = ["SCR diagnostic tools", "AdBlue system tools"]
                estimated_time = "4-6 hours"

            else:
                # Generic repair estimate
                part_cost = 200
                labor_hours = 3.0
                difficulty = "moderate"
                tools_required = ["Standard automotive tools"]
                estimated_time = "2-4 hours"

            # Calculate costs
            labor_cost = labor_hours * base_labor_rate
            subtotal = part_cost + labor_cost
            vat_amount = subtotal * vat_rate
            total_cost = subtotal + vat_amount

            # Add location context to description
            location_note = f"Estimated for {location} market. "
            if vat_rate > 0:
                location_note += f"Includes {int(vat_rate * 100)}% VAT. "
            location_note += f"Labor rate: €{base_labor_rate}/hour."

            estimate = RepairEstimate(
                part_name=repair_description,
                labor_hours=labor_hours,
                part_cost=part_cost,
                labor_rate=base_labor_rate,
                total_cost=round(total_cost, 2),
                description=f"{location_note} Parts: €{part_cost}, Labor: €{round(labor_cost, 2)}, VAT: €{round(vat_amount, 2)}",
                difficulty=difficulty,
                estimated_time=estimated_time,
                tools_required=tools_required,
                warranty="12-24 months parts, 12 months labor (varies by shop)",
            )

            return estimate

        except Exception as e:
            logger.error(f"Failed to estimate repair cost: {e}")
            return RepairEstimate(
                part_name=repair_description,
                labor_hours=2.0,
                part_cost=200.0,
                labor_rate=50.0,
                total_cost=300.0,
                description=f"Cost estimation error: {str(e)}",
                difficulty="moderate",
                estimated_time="Contact professional",
                tools_required=[],
                warranty="N/A",
            )

    async def generate_maintenance_schedule(
        self, vehicle_info: Dict[str, Any], current_mileage: int = 0, months_ahead: int = 12
    ) -> MaintenanceSchedule:
        """Generate a personalized maintenance schedule."""
        try:
            # Create basic maintenance items
            oil_change = MaintenanceItem(
                item="Oil Change",
                interval_miles=5000,
                interval_months=6,
                next_due=datetime.now() + timedelta(days=90),
                priority="high",
                description="Regular oil and filter change",
                overdue=False,
            )

            tire_rotation = MaintenanceItem(
                item="Tire Rotation",
                interval_miles=7500,
                interval_months=6,
                next_due=datetime.now() + timedelta(days=120),
                priority="medium",
                description="Rotate tires for even wear",
                overdue=False,
            )

            schedule = MaintenanceSchedule(
                vehicle_info=vehicle_info,
                current_mileage=current_mileage,
                items=[oil_change, tire_rotation],
                total_estimated_cost=150.0,
            )

            return schedule

        except Exception as e:
            logger.error(f"Failed to generate maintenance schedule: {e}")
            basic_item = MaintenanceItem(
                item="Schedule Error",
                interval_miles=5000,
                interval_months=6,
                next_due=datetime.now() + timedelta(days=90),
                priority="medium",
                description=f"Error generating schedule: {str(e)}",
                overdue=False,
            )

            return MaintenanceSchedule(
                vehicle_info=vehicle_info,
                current_mileage=current_mileage,
                items=[basic_item],
                total_estimated_cost=100.0,
            )

    async def get_european_regulations(
        self,
        regulation_type: str = "general",
        country: str = "Lithuania",
        vehicle_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get European automotive regulations and compliance requirements."""
        try:
            # EU Emission Standards
            emissions_info = {
                "current_standard": "Euro 6d",
                "applicable_since": "2020-09-01",
                "next_standard": "Euro 7 (expected 2025-2027)",
                "requirements": {
                    "petrol": {
                        "CO": "1.0 g/km",
                        "NOx": "0.06 g/km",
                        "HC": "0.10 g/km",
                        "PM": "0.0045 g/km (direct injection)",
                    },
                    "diesel": {
                        "CO": "0.50 g/km",
                        "NOx": "0.08 g/km",
                        "HC+NOx": "0.17 g/km",
                        "PM": "0.0045 g/km",
                    },
                },
            }

            # Country-specific inspection requirements
            inspection_requirements = {
                "Lithuania": {
                    "name": "Technical Inspection (Techninė apžiūra)",
                    "frequency": "Every 2 years (vehicles >4 years), Every year (vehicles >10 years)",
                    "cost_range": "€25-35",
                    "locations": "Regitra centers",
                    "required_documents": ["Registration certificate", "Insurance policy", "ID"],
                    "emission_test": "Required for vehicles >4 years",
                },
                "Germany": {
                    "name": "TÜV (Technischer Überwachungsverein)",
                    "frequency": "Every 2 years",
                    "cost_range": "€80-120",
                    "locations": "TÜV, DEKRA, GTÜ centers",
                    "required_documents": ["Vehicle registration", "Previous TÜV certificate"],
                    "emission_test": "AU (Abgasuntersuchung) required",
                },
                "France": {
                    "name": "Contrôle Technique",
                    "frequency": "Every 2 years (vehicles >4 years)",
                    "cost_range": "€70-100",
                    "locations": "Authorized CT centers",
                    "required_documents": ["Carte grise", "Insurance certificate"],
                    "emission_test": "Required for diesel vehicles",
                },
                "UK": {
                    "name": "MOT Test",
                    "frequency": "Every year (vehicles >3 years)",
                    "cost_range": "£54.85 maximum",
                    "locations": "Authorized MOT centers",
                    "required_documents": ["V5C (log book)", "Insurance certificate"],
                    "emission_test": "Required for all vehicles",
                },
            }

            # Vehicle-specific regulations
            vehicle_specific = {}
            if vehicle_info:
                vehicle_year = vehicle_info.get("year", 2020)
                vehicle_make = vehicle_info.get("make", "").lower()

                # Age-based requirements
                if vehicle_year and vehicle_year < 2005:
                    vehicle_specific["emission_zone_access"] = (
                        "May be restricted in low emission zones (LEZ/ULEZ)"
                    )
                elif vehicle_year and vehicle_year < 2015:
                    vehicle_specific["emission_zone_access"] = "Limited access to some city centers"
                else:
                    vehicle_specific["emission_zone_access"] = (
                        "Generally permitted in emission zones"
                    )

                # Brand-specific notes
                if vehicle_make in ["volkswagen", "audi", "skoda", "seat"]:
                    vehicle_specific["dieselgate_note"] = (
                        "Check for recalls related to emissions compliance (2015 dieselgate)"
                    )

                if vehicle_make in ["bmw", "mercedes"]:
                    vehicle_specific["adblue_note"] = (
                        "Many diesel models require AdBlue for Euro 6 compliance"
                    )

            # Low Emission Zones
            lez_info = {
                "concept": "Low Emission Zones restrict access for older, more polluting vehicles",
                "major_cities": [
                    "London (ULEZ)",
                    "Berlin (Umweltzone)",
                    "Paris (ZFE)",
                    "Amsterdam",
                    "Brussels",
                    "Milan",
                    "Madrid",
                ],
                "typical_restrictions": {
                    "diesel": "Euro 6 or newer typically required",
                    "petrol": "Euro 4 or newer typically required",
                    "penalties": "€80-180 fines for non-compliance",
                },
            }

            # AdBlue/SCR requirements
            adblue_info = {
                "required_for": "Most Euro 6 diesel vehicles (2015+)",
                "consumption": "Approximately 1L per 1000km",
                "cost": "€0.50-1.00 per liter",
                "availability": "Petrol stations, automotive stores",
                "legal_requirement": "Tampering with SCR system is illegal across EU",
            }

            # Build response based on regulation type
            if regulation_type.lower() == "emissions":
                result = {
                    "type": "EU Emissions Standards",
                    "emissions_standards": emissions_info,
                    "adblue_requirements": adblue_info,
                    "low_emission_zones": lez_info,
                }
            elif regulation_type.lower() == "inspection":
                country_inspection = inspection_requirements.get(
                    country, inspection_requirements["Lithuania"]
                )
                result = {
                    "type": "Vehicle Inspection Requirements",
                    "country": country,
                    "inspection_details": country_inspection,
                    "eu_standards": "All inspections must verify Euro emission compliance",
                }
            else:
                # General overview
                country_inspection = inspection_requirements.get(
                    country, inspection_requirements["Lithuania"]
                )
                result = {
                    "type": "European Automotive Regulations Overview",
                    "country": country,
                    "inspection_requirements": country_inspection,
                    "emission_standards": emissions_info,
                    "low_emission_zones": lez_info,
                    "vehicle_specific": vehicle_specific,
                    "adblue_requirements": adblue_info,
                    "important_notes": [
                        "Regulations vary by country - always check local requirements",
                        "Emission standards are EU-wide but enforcement varies",
                        "Low emission zones are becoming more common in city centers",
                        "Non-compliance can result in fines and access restrictions",
                    ],
                }

            return result

        except Exception as e:
            logger.error(f"Failed to get European regulations: {e}")
            return {
                "type": "Regulation Information Error",
                "error": str(e),
                "recommendation": "Consult official government sources for current regulations",
            }

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schemas for all available functions."""
        return [
            {
                "name": "diagnoseProblem",
                "description": "Diagnose car problems based on symptoms and vehicle information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symptoms": {
                            "type": "string",
                            "description": "Description of the car's symptoms and problems",
                        },
                        "vehicle_info": {
                            "type": "object",
                            "description": "Vehicle information (make, model, year, mileage, etc.)",
                            "properties": {
                                "make": {"type": "string"},
                                "model": {"type": "string"},
                                "year": {"type": "integer"},
                                "mileage": {"type": "integer"},
                                "engine_type": {"type": "string"},
                                "transmission": {"type": "string"},
                            },
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context from knowledge base",
                        },
                    },
                    "required": ["symptoms"],
                },
            },
            {
                "name": "estimateRepairCost",
                "description": "Estimate repair costs for specific repairs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repair_description": {
                            "type": "string",
                            "description": "Description of the repair needed",
                        },
                        "vehicle_info": {
                            "type": "object",
                            "description": "Vehicle information",
                            "properties": {
                                "make": {"type": "string"},
                                "model": {"type": "string"},
                                "year": {"type": "integer"},
                            },
                        },
                        "location": {
                            "type": "string",
                            "description": "Geographic location for cost estimation",
                            "default": "United States",
                        },
                    },
                    "required": ["repair_description"],
                },
            },
            {
                "name": "generateMaintenanceSchedule",
                "description": "Generate personalized maintenance schedule based on vehicle information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vehicle_info": {
                            "type": "object",
                            "description": "Complete vehicle information",
                            "properties": {
                                "make": {"type": "string"},
                                "model": {"type": "string"},
                                "year": {"type": "integer"},
                                "mileage": {"type": "integer"},
                                "engine_type": {"type": "string"},
                                "transmission": {"type": "string"},
                            },
                            "required": ["make", "model", "year"],
                        },
                        "current_mileage": {
                            "type": "integer",
                            "description": "Current vehicle mileage",
                            "default": 0,
                        },
                        "months_ahead": {
                            "type": "integer",
                            "description": "How many months to plan ahead",
                            "default": 12,
                        },
                    },
                    "required": ["vehicle_info"],
                },
            },
        ]
