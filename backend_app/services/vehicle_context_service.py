"""
Vehicle Context Service - Detects incomplete vehicle queries and prompts for vehicle information
"""

import re
from typing import Dict, Optional, Tuple

from backend_app.models.chat_models import VehicleInfo
from backend_app.utils.logger import logger


class VehicleContextService:
    """Service to detect vehicle-specific queries and manage vehicle information completeness."""

    def __init__(self):
        # Vehicle-specific keywords that indicate the user is asking about a specific vehicle
        # English keywords
        english_keywords = [
            # Problems/Issues
            "issues",
            "problems",
            "fault",
            "error",
            "trouble",
            "malfunction",
            "failure",
            "broken",
            "not working",
            "symptoms",
            "diagnostic",
            "repair",
            "fix",
            # Maintenance
            "maintenance",
            "service",
            "schedule",
            "oil change",
            "filter",
            "inspection",
            "tune-up",
            "timing belt",
            "brake pads",
            "tires",
            "fluid",
            "coolant",
            # Performance/Specifications
            "performance",
            "horsepower",
            "torque",
            "acceleration",
            "top speed",
            "fuel consumption",
            "mpg",
            "specifications",
            "engine specs",
            # Costs
            "cost",
            "price",
            "expensive",
            "cheap",
            "value",
            "worth",
            "budget",
            "estimate",
            "quote",
            "parts cost",
            "labor cost",
            # Buying/Selling
            "buy",
            "sell",
            "purchase",
            "worth buying",
            "good deal",
            "reliable",
            "common problems",
            "what to look for",
            "avoid",
            "recommended",
        ]

        # Lithuanian keywords
        lithuanian_keywords = [
            # Problems/Issues
            "problemos",
            "problema",
            "gedimai",
            "gedimas",
            "defektas",
            "defektai",
            "neveikia",
            "sugedo",
            "sugenda",
            "remontas",
            "pataisymas",
            "taisymas",
            # Maintenance
            "prieziura",
            "priežiūra",
            "aptarnavimas",
            "techninė",
            "techninis",
            "keitimas",
            "tikrinimas",
            "diagnostika",
            "patikra",
            # Costs
            "kaina",
            "kainos",
            "brangu",
            "pigu",
            "biudžetas",
            "išlaidos",
            "sąnaudos",
            "vertinimas",
            "pasiūlymas",
            "vertė",
            "verta",
            "pinigai",
            "eurai",
            "per metus",
            "metinis",
            "kainuoja",
            "kainuoti",
            "mokėti",
            "kaštai",
            "apsimoka",
            "brangus",
            "pigus",
            # Buying/Selling
            "pirkti",
            "parduoti",
            "pirkinys",
            "verta pirkti",
            "geras pasiūlymas",
            "patikimas",
            "dažnos problemos",
            "ką žiūrėti",
            "vengti",
            "rekomenduojamas",
        ]

        # Latvian keywords
        latvian_keywords = [
            # Problems/Issues
            "problēmas",
            "problēma",
            "bojājumi",
            "bojājums",
            "defekts",
            "defekti",
            "nestrādā",
            "salūzis",
            "remonts",
            "labošana",
            # Maintenance
            "apkope",
            "apkalpošana",
            "tehniskā",
            "tehniskais",
            "maiņa",
            "pārbaude",
            "diagnostika",
            # Costs
            "cena",
            "cenas",
            "dārgs",
            "lēts",
            "budžets",
            "izmaksas",
            "tēriņi",
            "novērtējums",
            "piedāvājums",
            "vērtība",
            "nauda",
            "eiro",
            "gadā",
            "aasta",
            # Buying/Selling
            "pirkt",
            "pārdot",
            "pirkums",
            "vērts pirkt",
            "labs piedāvājums",
            "uzticams",
            "biežas problēmas",
            "ko skatīties",
            "izvairīties",
            "ieteicams",
        ]

        # Estonian keywords
        estonian_keywords = [
            # Problems/Issues
            "probleemid",
            "probleem",
            "rikked",
            "rike",
            "defekt",
            "defektid",
            "ei tööta",
            "katki",
            "remont",
            "parandus",
            # Maintenance
            "hooldus",
            "teenindus",
            "tehniline",
            "vahetus",
            "kontroll",
            "diagnostika",
            # Costs
            "hind",
            "hinnad",
            "kallis",
            "odav",
            "eelarve",
            "kulud",
            "kulutused",
            "hinnang",
            "pakkumine",
            "väärtus",
            "raha",
            "euro",
            "aastas",
            "aasta",
            # Buying/Selling
            "osta",
            "müüa",
            "ost",
            "väärt ostma",
            "hea pakkumine",
            "usaldusväärne",
            "tavalised probleemid",
            "mida vaadata",
            "vältida",
            "soovitatud",
        ]

        # Polish keywords
        polish_keywords = [
            # Problems/Issues
            "problemy",
            "problem",
            "awarie",
            "awaria",
            "defekt",
            "defekty",
            "nie działa",
            "zepsuty",
            "naprawa",
            "naprawianie",
            # Maintenance
            "konserwacja",
            "serwis",
            "obsługa",
            "techniczny",
            "techniczna",
            "wymiana",
            "kontrola",
            "diagnostyka",
            # Costs
            "cena",
            "ceny",
            "drogi",
            "tani",
            "budżet",
            "koszty",
            "wydatki",
            "wycena",
            "oferta",
            "wartość",
            "pieniądze",
            "euro",
            "rocznie",
            "roku",
            # Buying/Selling
            "kupić",
            "sprzedać",
            "zakup",
            "warto kupić",
            "dobra oferta",
            "niezawodny",
            "częste problemy",
            "na co zwracać uwagę",
            "unikać",
            "polecany",
        ]

        # German keywords
        german_keywords = [
            # Problems/Issues
            "probleme",
            "problem",
            "defekte",
            "defekt",
            "fehler",
            "störung",
            "funktioniert nicht",
            "kaputt",
            "reparatur",
            "reparieren",
            # Maintenance
            "wartung",
            "service",
            "technisch",
            "technische",
            "wechsel",
            "prüfung",
            "diagnose",
            # Costs
            "preis",
            "preise",
            "teuer",
            "billig",
            "budget",
            "kosten",
            "ausgaben",
            "schätzung",
            "angebot",
            "wert",
            "geld",
            "euro",
            "jährlich",
            "jahr",
            # Buying/Selling
            "kaufen",
            "verkaufen",
            "kauf",
            "lohnt sich",
            "gutes angebot",
            "zuverlässig",
            "häufige probleme",
            "worauf achten",
            "vermeiden",
            "empfohlen",
        ]

        # French keywords
        french_keywords = [
            # Problems/Issues
            "problèmes",
            "problème",
            "défauts",
            "défaut",
            "panne",
            "dysfonctionnement",
            "ne fonctionne pas",
            "cassé",
            "réparation",
            "réparer",
            # Maintenance
            "entretien",
            "service",
            "technique",
            "révision",
            "changement",
            "contrôle",
            "diagnostic",
            # Costs
            "prix",
            "cher",
            "coûteux",
            "bon marché",
            "budget",
            "coûts",
            "dépenses",
            "estimation",
            "devis",
            "valeur",
            "argent",
            "euro",
            "par an",
            "annuel",
            # Buying/Selling
            "acheter",
            "vendre",
            "achat",
            "vaut la peine",
            "bonne affaire",
            "fiable",
            "problèmes courants",
            "quoi regarder",
            "éviter",
            "recommandé",
        ]

        # Combine all languages
        self.vehicle_specific_keywords = (
            english_keywords
            + lithuanian_keywords
            + latvian_keywords
            + estonian_keywords
            + polish_keywords
            + german_keywords
            + french_keywords
        )

        # Vehicle production year ranges for popular European models
        self.model_year_ranges = {
            # BMW Models
            "bmw": {
                # M Series
                "m2": (2016, 2024),  # F22/F87 (2016-2020), G87 (2021+)
                "m3": (1985, 2024),  # Multiple generations E30, E36, E46, E90, F80, G80
                "m4": (2014, 2024),  # F82/F83 (2014-2020), G82/G83 (2021+)
                "m5": (1985, 2024),  # Multiple generations E28, E34, E39, E60, F10, F90
                "m6": (1983, 2020),  # E24, E63/E64, F06/F12/F13
                "m8": (1989, 1999, 2018, 2024),  # E31 (1989-1999), F91/F92/F93 (2018+)
                # X Series
                "x1": (2009, 2024),
                "x2": (2017, 2024),
                "x3": (2003, 2024),
                "x4": (2014, 2024),
                "x5": (1999, 2024),
                "x6": (2008, 2024),
                "x7": (2018, 2024),
                # Z Series
                "z3": (1995, 2002),
                "z4": (2002, 2008, 2009, 2016, 2017, 2024),  # E85/E86, E89, G29
                "z8": (1999, 2003),
                # Regular Series (major generations)
                "3 series": (1975, 2024),  # General 3 series
                "e30": (1982, 1994),
                "e36": (1990, 2000),
                "e46": (1998, 2006),
                "e90": (2005, 2013),
                "f30": (2011, 2019),
                "g20": (2018, 2024),
                "5 series": (1972, 2024),  # General 5 series
                "e39": (1995, 2004),
                "e60": (2003, 2010),
                "f10": (2009, 2017),
                "g30": (2016, 2024),
                "7 series": (1977, 2024),
            },
            # Mercedes Models
            "mercedes": {
                # AMG Models
                "c55": (2004, 2007),  # W203 C55 AMG
                "c63": (2007, 2024),  # W204, W205, W206
                "c32": (2001, 2004),  # W203 C32 AMG
                "c43": (1997, 2000, 2016, 2024),  # W202, current W205/W206
                # Regular C-Class Models
                "c350": (2005, 2015),  # W204, W205 C350
                "c300": (2007, 2024),  # W204, W205, W206
                "c250": (2011, 2024),  # W204, W205, W206
                "c200": (1993, 2024),  # Multiple generations
                "c220": (1993, 2024),  # Multiple generations
                "e55": (1993, 1997, 2002, 2006),  # W124, W210, W211
                "e63": (2006, 2024),  # W211, W212, W213
                "e43": (2016, 2024),  # Active generation
                "s55": (1998, 2002, 2002, 2006),  # W220
                "s63": (2005, 2024),  # W221, W222, W223
                "s65": (2003, 2020),  # W220, W221, W222
                # Regular Models
                "c-class": (1993, 2024),
                "e-class": (1985, 2024),
                "s-class": (1972, 2024),
                # Chassis Codes
                "w203": (2000, 2007),  # C-Class
                "w204": (2007, 2014),  # C-Class
                "w205": (2014, 2021),  # C-Class
                "w206": (2021, 2024),  # C-Class
                "w210": (1995, 2003),  # E-Class
                "w211": (2002, 2009),  # E-Class
                "w212": (2009, 2016),  # E-Class
                "w213": (2016, 2024),  # E-Class
            },
            # Audi Models
            "audi": {
                "a3": (1996, 2024),
                "a4": (1994, 2024),
                "a5": (2007, 2024),
                "a6": (1994, 2024),
                "a7": (2010, 2024),
                "a8": (1994, 2024),
                "q3": (2011, 2024),
                "q5": (2008, 2024),
                "q7": (2005, 2024),
                "q8": (2018, 2024),
                "s3": (1999, 2024),
                "s4": (1991, 2024),
                "s5": (2007, 2024),
                "s6": (1994, 2024),
                "s7": (2012, 2024),
                "s8": (1996, 2024),
                "rs3": (2011, 2024),
                "rs4": (1999, 2024),
                "rs5": (2010, 2024),
                "rs6": (2002, 2024),
                "rs7": (2013, 2024),
                "rs8": (2019, 2024),
                "tt": (1998, 2024),
                "r8": (2006, 2024),
            },
        }

        # European car brands and their common name variations
        self.european_brands = [
            # German
            "bmw",
            "mercedes-benz",
            "mercedes",
            "mb",
            "audi",
            "volkswagen",
            "vw",
            "porsche",
            "opel",
            # French
            "peugeot",
            "renault",
            "citroën",
            "citroen",
            "ds",
            # Italian
            "fiat",
            "ferrari",
            "lamborghini",
            "maserati",
            "alfa romeo",
            "alfa",
            "lancia",
            # Swedish
            "volvo",
            "saab",
            # British
            "jaguar",
            "land rover",
            "range rover",
            "aston martin",
            "lotus",
            "mini",
            # Spanish
            "seat",
            "cupra",
            # Czech
            "skoda",
            "škoda",
            # Romanian
            "dacia",
            # Others popular in Europe
            "ford",
            "toyota",
            "honda",
            "nissan",
            "mazda",
            "hyundai",
            "kia",
        ]

        # Model patterns that indicate specific vehicle queries
        self.model_patterns = [
            # Mercedes models
            r"\bc[0-9]{2,3}\b",
            r"\be[0-9]{2,3}\b",
            r"\bs[0-9]{2,3}\b",  # C55, E63, S500, C350
            r"\bw[0-9]{3}\b",
            r"\br[0-9]{3}\b",  # W204, R230
            r"\bamg\b",
            r"\bgl[eck]?\b",
            r"\bsl[rk]?\b",  # AMG, GLE, SLR
            # BMW models
            r"\b[0-9]{1}series\b",
            r"\bx[0-9]{1}\b",
            r"\bz[0-9]{1}\b",  # 3series, X5, Z4
            r"\bm[0-9]{1}\b",
            r"\bi[0-9]{1}\b",  # M3, i8
            r"\be[0-9]{2}\b",
            r"\bf[0-9]{2}\b",
            r"\bg[0-9]{2}\b",  # E46, F30, G20
            # Audi models
            r"\ba[0-9]{1}\b",
            r"\bq[0-9]{1}\b",
            r"\btt\b",
            r"\br8\b",  # A4, Q7, TT, R8
            r"\bs[0-9]{1}\b",
            r"\brs[0-9]{1}\b",  # S4, RS6
            # VW models
            r"\bgolf\b",
            r"\bpolo\b",
            r"\bpassat\b",
            r"\bjetta\b",
            r"\btiguan\b",
            r"\btouareg\b",
            r"\bgti\b",
            r"\br32\b",
            # Other common European models
            r"\bfocus\b",
            r"\bfiesta\b",
            r"\bmondeo\b",  # Ford
            r"\b[0-9]{3}\b",  # Peugeot/BMW 3-digit models (206, 320i, etc.)
            r"\boctavia\b",
            r"\bfabia\b",
            r"\bsuperb\b",  # Skoda
            r"\bclio\b",
            r"\bmegane\b",
            r"\blaguna\b",  # Renault
            r"\b500\b",
            r"\bpunto\b",
            r"\bpanda\b",  # Fiat
        ]

    def detect_vehicle_specific_query(self, message: str) -> bool:
        """
        Detect if a message is asking about a specific vehicle.

        Args:
            message: User's message

        Returns:
            True if the query is vehicle-specific
        """
        message_lower = message.lower()

        # Check if message contains a car brand
        has_brand = any(brand in message_lower for brand in self.european_brands)

        # Check if message contains vehicle-specific keywords
        has_vehicle_keywords = any(
            keyword in message_lower for keyword in self.vehicle_specific_keywords
        )

        # Check if message contains model patterns
        has_model_pattern = any(
            re.search(pattern, message_lower) for pattern in self.model_patterns
        )

        # Conservative vehicle-specific detection logic
        # Only consider vehicle-specific if it has BOTH brand AND model pattern
        # This allows generic brand queries to use RAG first
        is_vehicle_specific = has_brand and has_model_pattern

        # EXCEPTION: If query explicitly asks for vehicle-specific info (buying, reliability, etc.)
        # without mentioning a specific model, then prompt for details
        buying_keywords = [
            "buy",
            "purchase",
            "worth buying",
            "reliable",
            "avoid",
            "generation",
            "year",
            "model year",
        ]
        has_buying_intent = any(keyword in message_lower for keyword in buying_keywords)

        if has_brand and has_buying_intent and not has_model_pattern:
            is_vehicle_specific = True

        logger.info(
            f"Vehicle specificity check: brand={has_brand}, keywords={has_vehicle_keywords}, model={has_model_pattern}, buying={has_buying_intent}, result={is_vehicle_specific}"
        )

        return is_vehicle_specific

    def analyze_vehicle_info_completeness(
        self, vehicle_info: Optional[VehicleInfo], message: str
    ) -> Dict[str, any]:
        """
        Analyze how complete the vehicle information is for a specific query.

        Args:
            vehicle_info: Current vehicle information
            message: User's message

        Returns:
            Dictionary with completeness analysis
        """
        # Extract vehicle info from message if not provided
        extracted_info = self._extract_vehicle_from_message(message)

        # Combine provided and extracted info
        combined_info = self._combine_vehicle_info(vehicle_info, extracted_info)

        # Determine what's missing
        missing_fields = []
        critical_missing = []

        if not combined_info.get("make"):
            missing_fields.append("make")
            critical_missing.append("vehicle brand (Mercedes, BMW, Audi, etc.)")

        if not combined_info.get("model"):
            missing_fields.append("model")
            critical_missing.append("model (C55, M3, A4, etc.)")

        if not combined_info.get("year"):
            missing_fields.append("year")
            critical_missing.append("year")

        # Determine if year is critical based on query type
        year_critical = any(
            word in message.lower()
            for word in [
                "buy",
                "purchase",
                "worth",
                "reliable",
                "problems",
                "issues",
                "maintenance",
                "cost",
                "value",
                "avoid",
                "generation",
            ]
        )

        return {
            "is_complete": len(critical_missing) == 0,
            "missing_fields": missing_fields,
            "critical_missing": critical_missing,
            "extracted_info": extracted_info,
            "combined_info": combined_info,
            "year_critical": year_critical,
            "completeness_score": self._calculate_completeness_score(combined_info, year_critical),
        }

    def generate_vehicle_info_prompt(
        self, completeness_analysis: Dict[str, any], message: str
    ) -> str:
        """
        Generate a helpful prompt asking the user to provide missing vehicle information.

        Args:
            completeness_analysis: Result from analyze_vehicle_info_completeness
            message: Original user message

        Returns:
            Formatted prompt for the user
        """
        critical_missing = completeness_analysis["critical_missing"]
        extracted_info = completeness_analysis["extracted_info"]

        # Base prompt
        prompt = "🚗 **I'd love to help you with specific information!**\n\n"

        # Acknowledge what we detected
        if extracted_info.get("make") or extracted_info.get("model"):
            detected_parts = []
            if extracted_info.get("make"):
                detected_parts.append(f"**{extracted_info['make']}**")
            if extracted_info.get("model"):
                detected_parts.append(f"**{extracted_info['model']}**")
            if extracted_info.get("year"):
                detected_parts.append(f"**{extracted_info['year']}**")

            prompt += f"I can see you're asking about: {' '.join(detected_parts)}\n\n"

        # Explain why we need more info
        if "generation" in message.lower() or "version" in message.lower():
            prompt += (
                "Since different model years can have **significantly different** characteristics, "
            )
        else:
            prompt += "To provide you with **accurate, model-specific** information, "

        prompt += "could you please fill in the vehicle details on the left sidebar?\n\n"

        # Specific missing fields
        prompt += "**I need:**\n"
        for missing in critical_missing:
            prompt += f"• {missing.title()}\n"

        # Add helpful context based on query type
        if any(word in message.lower() for word in ["issues", "problems", "reliable"]):
            prompt += "\n💡 **Why this matters:** Different model years can have completely different common issues and reliability patterns."
        elif any(word in message.lower() for word in ["cost", "price", "maintenance"]):
            prompt += "\n💡 **Why this matters:** Maintenance costs and procedures vary significantly between model years and engine variants."
        elif any(word in message.lower() for word in ["buy", "purchase", "worth"]):
            prompt += "\n💡 **Why this matters:** Each model year has different features, known issues, and market values to consider."

        # Call to action
        prompt += "\n\n**Once you fill that in, I'll give you detailed, year-specific advice!** 🎯"

        return prompt

    def should_prompt_for_vehicle_info(
        self, message: str, vehicle_info: Optional[VehicleInfo]
    ) -> Tuple[bool, Optional[str]]:
        """
        Main function to determine if we should prompt for vehicle information.

        Args:
            message: User's message
            vehicle_info: Current vehicle information

        Returns:
            Tuple of (should_prompt, prompt_message)
        """
        # First check if the query is vehicle-specific
        if not self.detect_vehicle_specific_query(message):
            return False, None

        # Check for vehicle model-year validation issues FIRST
        if vehicle_info and vehicle_info.make and vehicle_info.model and vehicle_info.year:
            is_valid_combination, validation_error = self.validate_vehicle_model_year_combination(
                vehicle_info.make, vehicle_info.model, vehicle_info.year
            )
            if not is_valid_combination:
                logger.warning(
                    f"Invalid vehicle combination detected: {vehicle_info.make} {vehicle_info.model} {vehicle_info.year}"
                )

                # Generate clarification prompt
                clarification_prompt = "🚗 **Vehicle Information Clarification Needed**\n\n"
                clarification_prompt += validation_error + "\n\n"
                clarification_prompt += "**Please update your vehicle information in the left sidebar with the correct details.** 🔧\n\n"
                clarification_prompt += "Once you provide the correct vehicle information, I'll give you accurate, model-specific advice! 🎯"

                return True, clarification_prompt

        # Then analyze completeness
        analysis = self.analyze_vehicle_info_completeness(vehicle_info, message)

        # If completeness score is too low, prompt for info
        logger.info(
            f"Completeness analysis: score={analysis['completeness_score']}, missing={analysis['critical_missing']}"
        )

        if analysis["completeness_score"] < 0.8:  # Threshold for prompting (increased)
            prompt = self.generate_vehicle_info_prompt(analysis, message)
            return True, prompt

        return False, None

    def _extract_vehicle_from_message(self, message: str) -> Dict[str, any]:
        """Extract vehicle information from message text."""
        message_lower = message.lower()
        extracted = {}

        # Extract brand
        for brand in self.european_brands:
            if brand in message_lower:
                if brand == "vw":
                    extracted["make"] = "Volkswagen"
                elif brand == "mercedes":
                    extracted["make"] = "Mercedes-Benz"
                else:
                    extracted["make"] = brand.title()
                break

        # Extract year
        year_match = re.search(r"\b(19[8-9]\d|20[0-3]\d)\b", message)
        if year_match:
            extracted["year"] = int(year_match.group(1))

        # Extract model (simplified)
        for pattern in self.model_patterns:
            match = re.search(pattern, message_lower)
            if match:
                extracted["model"] = match.group(0).upper()
                break

        # Special handling for common model patterns
        if "c55" in message_lower:
            extracted["model"] = "C55 AMG"
        elif "c63" in message_lower:
            extracted["model"] = "C63 AMG"
        elif "c350" in message_lower:
            extracted["model"] = "C350"
        elif "c300" in message_lower:
            extracted["model"] = "C300"
        elif "c250" in message_lower:
            extracted["model"] = "C250"
        elif "m3" in message_lower:
            extracted["model"] = "M3"
        elif "golf" in message_lower:
            extracted["model"] = "Golf"

        return extracted

    def validate_vehicle_model_year_combination(
        self, make: str, model: str, year: int
    ) -> tuple[bool, str]:
        """
        Validate if a vehicle make/model/year combination is technically possible.
        Returns (is_valid, message)
        """
        # Normalize make to lowercase for consistent lookup (data uses lowercase keys)
        make_normalized = make.lower().strip()

        # Handle common variations and create the lookup key
        if make_normalized in ["mercedes", "mercedes-benz"]:
            make_lookup = "mercedes"
            make_display = "Mercedes-Benz"
        elif make_normalized in ["vw", "volkswagen"]:
            make_lookup = "volkswagen"
            make_display = "Volkswagen"
        elif make_normalized == "bmw":
            make_lookup = "bmw"
            make_display = "BMW"
        else:
            make_lookup = make_normalized
            make_display = make.title()

        # Check if we have year range data for this make
        if make_lookup not in self.model_year_ranges:
            # We don't have data for this make, assume valid
            return True, ""

        make_data = self.model_year_ranges[make_lookup]
        model_lower = model.lower()

        # Check if model exists for this make
        if model_lower not in make_data:
            # We don't have data for this model, assume valid
            return True, ""

        model_data = make_data[model_lower]

        # Handle multiple production periods (like BMW M8: 1989-1999, 2018-2024)
        if isinstance(model_data, tuple) and len(model_data) == 4:
            # Two production periods: (start1, end1, start2, end2)
            start1, end1, start2, end2 = model_data
            if (start1 <= year <= end1) or (start2 <= year <= end2):
                return True, ""

            # If we get here, year doesn't match any period
            periods_str = f"{start1}-{end1}, {start2}-{end2}"

            # Suggest closest valid years
            all_years = list(range(start1, end1 + 1)) + list(range(start2, end2 + 1))
            closest_year = min(all_years, key=lambda x: abs(x - year))
            suggestion = f"Did you mean {make_display} {model.upper()} {closest_year}?"

            return (
                False,
                f"⚠️ The {make_display} {model.upper()} was produced during: {periods_str}, but you entered {year}. {suggestion}",
            )

        else:
            # Single production period
            start_year, end_year = model_data
            if start_year <= year <= end_year:
                return True, ""

            # Generate helpful suggestion
            if year < start_year:
                # Year too early - suggest earliest year or similar model
                suggestion = f"Did you mean {make_display} {model.upper()} {start_year}"
                if year <= start_year - 5:
                    # Suggest alternative model that might have existed
                    suggestions = []
                    for other_model, other_data in make_data.items():
                        if isinstance(other_data, tuple) and len(other_data) == 2:
                            other_start, other_end = other_data
                        elif isinstance(other_data, tuple) and len(other_data) == 4:
                            # Handle multiple periods
                            other_start = min(other_data[0], other_data[2])
                            other_end = max(other_data[1], other_data[3])
                        else:
                            continue

                        if other_start <= year <= other_end and other_model != model_lower:
                            suggestions.append(f"{make_display} {other_model.upper()} {year}")

                    if suggestions:
                        suggestion += f" or {suggestions[0]}"

                suggestion += "?"
            else:
                # Year too late - suggest latest year
                suggestion = f"Did you mean {make_display} {model.upper()} {end_year}?"

            return (
                False,
                f"⚠️ The {make_display} {model.upper()} was produced from {start_year} to {end_year}, but you entered {year}. {suggestion}",
            )

    def _combine_vehicle_info(
        self, provided_info: Optional[VehicleInfo], extracted_info: Dict[str, any]
    ) -> Dict[str, any]:
        """Combine provided vehicle info with extracted info."""
        combined = extracted_info.copy()

        if provided_info:
            if provided_info.make:
                combined["make"] = provided_info.make
            if provided_info.model:
                combined["model"] = provided_info.model
            if provided_info.year:
                combined["year"] = provided_info.year
            if provided_info.mileage:
                combined["mileage"] = provided_info.mileage

        return combined

    def _calculate_completeness_score(
        self, vehicle_info: Dict[str, any], year_critical: bool
    ) -> float:
        """Calculate a completeness score from 0.0 to 1.0."""
        score = 0.0

        # Brand is always critical (40% weight)
        if vehicle_info.get("make"):
            score += 0.4

        # Model is highly important (30% weight)
        if vehicle_info.get("model"):
            score += 0.3

        # Year importance depends on query type
        if vehicle_info.get("year"):
            score += 0.3 if year_critical else 0.2

        # Mileage is nice to have (10% weight)
        if vehicle_info.get("mileage"):
            score += 0.1

        return min(score, 1.0)
