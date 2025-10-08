"""
Comprehensive European car troubleshooting knowledge base for RAG system
Specialized for European markets, with focus on Lithuania and Baltic region
Complete automotive repair and diagnostic information with European context
"""

from datetime import datetime
from typing import List

# Local utility functions


def deduplicate_sources(sources):
    """Remove duplicate sources based on content similarity."""
    if not sources:
        return []

    unique_sources = []
    seen_content = set()

    for source in sources:
        # Create a simplified version for comparison
        if hasattr(source, "content"):
            content_key = source.content.lower().strip()[:100]  # First 100 chars
        elif hasattr(source, "page_content"):
            content_key = source.page_content.lower().strip()[:100]
        else:
            content_key = str(source).lower().strip()[:100]

        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_sources.append(source)

    return unique_sources


class KnowledgeDocument:
    """Knowledge base document structure."""

    def __init__(
        self,
        id: str,
        title: str,
        content: str,
        category: str,
        tags: List[str],
        source: str,
        last_updated: datetime = None,
    ):
        self.id = id
        self.title = title
        self.content = content
        self.category = category
        self.tags = tags
        self.source = source
        self.last_updated = last_updated if last_updated is not None else datetime.now()


def is_duplicate_document(
    new_doc: KnowledgeDocument, existing_docs: List[KnowledgeDocument]
) -> bool:
    """Check if a new document is a duplicate of existing documents based on title and content similarity."""
    for doc in existing_docs:
        if new_doc.title == doc.title or new_doc.content == doc.content:
            return True
    return False


def validate_document_content(new_doc: KnowledgeDocument) -> bool:
    """Validate the content of a new document to ensure it is relevant and accurate."""
    # Example validation logic: Check if the content contains essential automotive keywords
    essential_keywords = ["engine", "battery", "diagnostic", "repair", "maintenance"]
    content_lower = new_doc.content.lower()

    # Ensure the content contains at least one essential keyword
    if any(keyword in content_lower for keyword in essential_keywords):
        return True

    # Log invalid content for review
    print(f"Invalid document content: {new_doc.title}")
    return False


# European-specific car knowledge base documents
EUROPEAN_KNOWLEDGE_BASE = [
    KnowledgeDocument(
        id="european-engine-wont-start",
        title="Engine Won't Start - European Vehicles",
        content="""
SYMPTOMS: Engine cranks but won't start, no ignition, clicking sounds, dashboard lights dimming.

EUROPEAN CONSIDERATIONS:
- Cold climate starting issues (especially in Lithuania, Latvia, Estonia)
- Diesel engine glow plug problems common in European cars
- AdBlue/DEF system issues in modern diesel vehicles
- Stop-start system malfunctions

COMMON CAUSES:
1. Dead Battery: Most common in cold European winters. Battery voltage below 12.4V.
2. Glow Plug Issues (Diesel): Failed glow plugs prevent cold starts in diesel engines
3. AdBlue System (Diesel): Empty AdBlue tank or system fault prevents starting
4. Fuel System Issues: Diesel fuel gelling in extreme cold, contaminated fuel
5. Immobilizer/Security: European vehicles have sophisticated anti-theft systems

EUROPEAN BRANDS SPECIFICS:
- BMW: Common N47/N57 diesel timing chain issues
- Mercedes: CDI diesel injector problems
- Audi/VW: TDI emissions system faults, timing belt issues
- Volvo: D5 diesel swirl flap problems
- Peugeot/Citroën: HDI diesel fuel system issues

DIAGNOSTIC STEPS:
1. Check battery voltage and terminals
2. Test glow plugs (diesel vehicles)
3. Check AdBlue level and system (modern diesels)
4. Scan for fault codes with European diagnostic tools
5. Test fuel pressure and quality

ESTIMATED COSTS (EUR):
- Battery replacement: €80-150
- Glow plug replacement: €120-300 (set of 4)
- AdBlue system repair: €200-800
- Fuel pump replacement: €300-900
- Immobilizer repair: €150-500

WINTER PREPARATION:
- Use engine block heater in extreme cold
- Winter-grade diesel fuel additives
- Battery maintenance critical in Baltic climate

SAFETY: Diesel fuel systems under high pressure. Professional diagnosis recommended.
        """,
        category="Engine",
        tags=["starting", "engine", "diesel", "european", "winter", "glow-plugs"],
        source="European Automotive Repair Guide",
    ),
    KnowledgeDocument(
        id="european-dpf-regeneration",
        title="DPF (Diesel Particulate Filter) Issues",
        content="""
SYMPTOMS: Reduced power, excessive fuel consumption, DPF warning light, limp mode.

EUROPEAN CONTEXT:
- Mandatory on all Euro 5+ diesel vehicles
- Common issue in city driving conditions
- Especially problematic in Eastern European urban areas
- Strict emissions regulations require proper function

DPF REGENERATION PROCESS:
1. Passive: Occurs during highway driving (over 60 km/h)
2. Active: ECU initiates cleaning cycle
3. Forced: Manual regeneration via diagnostic tool
4. Service: Professional cleaning or replacement

COMMON CAUSES:
1. Short Journey Driving: City driving prevents proper regeneration
2. Poor Quality Diesel: High sulfur content clogs filter faster
3. Engine Oil Contamination: Wrong oil grade affects regeneration
4. Sensor Failures: Pressure sensors give incorrect readings
5. Turbocharger Issues: Reduced exhaust flow affects regeneration

EUROPEAN BRAND SPECIFICS:
- BMW: N47/N57 engines - DPF clogs every 80,000-120,000 km
- Mercedes: OM651 - pressure sensor failures common
- VW/Audi TDI: DPF regeneration software issues
- Peugeot/Citroën HDI: DPF problems around 100,000 km
- Ford TDCi: EGR valve affects DPF function

PREVENTION:
- Regular highway driving (20+ minutes at 70+ km/h monthly)
- Use low-ash engine oil (ACEA C1/C2/C3)
- Quality Euro 5/6 diesel fuel
- Don't ignore DPF warning lights

REPAIR OPTIONS:
1. Forced Regeneration: €50-100
2. DPF Cleaning: €200-400
3. DPF Replacement: €1,500-3,500
4. Complete Service: €300-600

LEGAL CONSIDERATIONS:
- DPF removal illegal in EU (including Lithuania)
- MOT/Technical inspection failure if removed
- Emissions compliance mandatory
- Heavy fines for tampering with emissions equipment

MAINTENANCE SCHEDULE:
- DPF inspection: Every 20,000 km
- Forced regeneration: As needed
- Complete service: Every 80,000-120,000 km
        """,
        category="Emissions",
        tags=["dpf", "diesel", "emissions", "regeneration", "european", "euro5", "euro6"],
        source="European Emissions Compliance Manual",
    ),
    KnowledgeDocument(
        id="european-brake-systems",
        title="European Brake Systems and Regulations",
        content="""
SYMPTOMS: Squealing, grinding, soft pedal, pulling, vibration, warning lights.

EUROPEAN BRAKE STANDARDS:
- ECE R13 brake performance standards
- ESP/ESC mandatory on EU vehicles since 2014
- Advanced brake assist systems standard
- Brake fluid DOT 4+ required for most European vehicles

BRAKE PAD WEAR INDICATORS:
- Electronic wear sensors standard on premium brands
- Acoustic warning indicators common
- Visual inspection through wheel spokes
- Typical lifespan: 20,000-50,000 km (European driving conditions)

EUROPEAN BRAND SPECIFICS:
- BMW: Brake wear sensors integrated into pads
- Mercedes: SBC (Sensotronic) brake system issues (2003-2009)
- Audi: Ceramic brake options on performance models
- Volvo: Advanced brake assist with pedestrian detection
- Peugeot/Citroën: Bendix brake system components

BRAKE FLUID SPECIFICATIONS:
- DOT 4: Standard for most European vehicles
- DOT 5.1: Required for performance/sports cars
- Low viscosity fluids for ESP systems
- Hygroscopic properties critical in humid climates

WINTER CONSIDERATIONS:
- Road salt corrosion on brake components
- Frozen brake lines in extreme cold
- Snow/ice affecting brake performance
- Brake fluid freezing point considerations

MAINTENANCE SCHEDULE (European Standards):
- Brake inspection: Every 15,000 km or annually
- Brake fluid change: Every 2-3 years (mandatory for technical inspection)
- Brake pad replacement: 25,000-50,000 km
- Brake disc replacement: 50,000-100,000 km

TECHNICAL INSPECTION REQUIREMENTS:
- Lithuania: Brake test on roller dynamometer
- Germany (TÜV): Strict brake performance standards
- UK (MOT): Annual brake efficiency test
- Scandinavian countries: Enhanced winter brake testing

ESTIMATED COSTS (EUR):
- Brake pad replacement (front): €100-250
- Brake disc replacement (pair): €150-400
- Brake fluid service: €50-80
- Complete brake service (per axle): €200-500
- ESP system repair: €300-1,200

SAFETY WARNINGS:
- Grinding noise: Immediate attention required
- Soft pedal: Brake fluid leak or air in system
- Electronic warnings: Scan for fault codes
- Pull to one side: Uneven brake wear or stuck caliper

ENVIRONMENTAL CONSIDERATIONS:
- Brake dust regulations in urban areas
- Proper disposal of brake fluid and components
- Low-copper brake pads required in some regions
        """,
        category="Brakes",
        tags=["brakes", "european", "safety", "regulations", "technical-inspection", "esp"],
        source="European Brake Systems Manual",
    ),
    KnowledgeDocument(
        id="european-tire-regulations",
        title="European Tire Regulations and Maintenance",
        content="""
EUROPEAN TIRE STANDARDS:
- ECE markings mandatory (E1, E2, etc.)
- EU tire labeling for fuel efficiency, wet grip, noise
- Minimum tread depth: 1.6mm (3mm recommended for winter)
- M+S or 3PMSF markings for winter tires

TIRE PRESSURE MONITORING (TPMS):
- Mandatory on all EU vehicles since 2014
- Direct or indirect systems
- Warning at 25% pressure loss
- Reset required after tire changes

WINTER TIRE REQUIREMENTS:
- Lithuania: Winter tires mandatory Nov 10 - Apr 10
- Latvia: Winter tires mandatory Dec 1 - Mar 1
- Estonia: Winter tires mandatory Dec 1 - Mar 1
- Germany: Situational requirement (bei winterlichen Straßenverhältnissen)
- Scandinavia: Studded tires allowed with restrictions

TIRE ROTATION PATTERNS:
- Front-wheel drive: Front to rear, cross rear
- Rear-wheel drive: Rear to front, cross front
- All-wheel drive: Straight rotation or diagonal
- Directional tires: Front to rear only same side

EUROPEAN BRAND PREFERENCES:
- Premium: Michelin, Continental, Bridgestone
- Mid-range: Goodyear, Pirelli, Dunlop
- Budget: Nexen, Hankook, Nokian (excellent for winter)
- Local: Barum (Central Europe), Vredestein (Netherlands)

SPEED RATINGS COMMON IN EUROPE:
- H (210 km/h): Standard for family cars
- V (240 km/h): Performance vehicles
- W (270 km/h): High-performance vehicles
- Y (300 km/h): Sports cars

MAINTENANCE SCHEDULE:
- Pressure check: Monthly (when cold)
- Rotation: Every 8,000-10,000 km
- Alignment check: Annually or after impact
- Replacement: When tread reaches 2-3mm

SEASONAL CONSIDERATIONS:
- Summer tires: Apr-Oct (depending on region)
- Winter tires: Oct/Nov-Mar/Apr
- All-season: Compromise option (not ideal for harsh winters)
- Storage: Cool, dry, dark location

COST ESTIMATES (EUR):
- Budget tires: €50-80 per tire
- Mid-range tires: €80-150 per tire
- Premium tires: €150-300 per tire
- Winter tire set (4): €200-800
- TPMS sensor: €50-100 each
- Wheel alignment: €50-100

TECHNICAL INSPECTION:
- Tread depth measurement mandatory
- TPMS function test required
- Sidewall damage inspection
- Correct tire specifications verification

ENVIRONMENTAL IMPACT:
- EU tire recycling programs
- Low rolling resistance requirements
- Noise regulations in urban areas
- Eco-friendly tire compounds promoted
        """,
        category="Tires",
        tags=["tires", "european", "winter", "tpms", "regulations", "technical-inspection"],
        source="European Tire Regulations Guide",
    ),
    KnowledgeDocument(
        id="european-engine-oil-standards",
        title="European Engine Oil Standards and Specifications",
        content="""
EUROPEAN OIL SPECIFICATIONS:
- ACEA (European standards): A/B, C, E series
- Manufacturer specific: VW 504.00, BMW LL-04, MB 229.51
- API compatibility ratings
- Viscosity grades for European climate

ACEA CLASSIFICATIONS:
- A/B: Gasoline and light diesel engines
- C: Compatible with catalysts (Low SAPS)
- E: Heavy duty diesel engines
- Lower numbers = older technology, higher = newer

POPULAR EUROPEAN SPECIFICATIONS:
- ACEA C3: Most common for modern European cars
- VW 504.00/507.00: Longlife oils for VAG group
- BMW LL-01/LL-04: Longlife specifications
- Mercedes 229.31/229.51: MB Approval oils

VISCOSITY FOR EUROPEAN CLIMATE:
- 0W-30: Extreme cold conditions (Scandinavia, Baltic)
- 5W-30: Most common for year-round use
- 5W-40: Older engines, high mileage vehicles
- 0W-20: Modern fuel-efficient engines

BRAND-SPECIFIC REQUIREMENTS:
- BMW: LL-04 for diesel, LL-01 for gasoline
- Mercedes: 229.51 for modern engines
- VW/Audi: 504.00/507.00 for TDI engines
- Volvo: VCC RBS0-2AE for modern engines
- PSA (Peugeot/Citroën): PSA B71 2290/2300

OIL CHANGE INTERVALS:
- Variable service: 15,000-30,000 km (longlife oils)
- Fixed service: 10,000-15,000 km
- Severe conditions: 7,500-10,000 km
- City driving: Shorter intervals recommended

QUALITY INDICATORS:
- Golden/amber color when fresh
- Black color indicates contamination
- Metal particles: Engine wear concern
- Milky appearance: Coolant contamination
- Thick consistency: Oxidation or contamination

EUROPEAN BRAND OILS:
- Premium: Castrol, Mobil 1, Shell Helix
- Quality: Total, Elf, Motul
- Budget: Mannol, Liqui Moly, Fuchs
- Dealer: Original manufacturer oils

COST ESTIMATES (EUR):
- Basic oil change: €40-80
- Premium oil change: €80-150
- Longlife oil service: €100-200
- Oil analysis: €30-50
- Filter replacement: €15-40

ENVIRONMENTAL CONSIDERATIONS:
- Used oil recycling mandatory in EU
- Low emission oil formulations
- Extended drain intervals reduce waste
- Biodegradable oil options available

WINTER CONSIDERATIONS:
- 0W grades essential for cold starts
- Block heaters reduce oil stress
- Synthetic oils flow better in cold
- Frequent short trips require shorter intervals

TECHNICAL INSPECTION:
- Oil level check required
- Oil condition assessment
- Leak inspection mandatory
- Service history verification

DIAGNOSTIC INDICATORS:
- Oil pressure warning: Immediate stop required
- Oil life monitoring systems: Follow recommendations
- Electronic service reminders: Reset after service
- Oil quality sensors: Available on premium vehicles
        """,
        category="Engine",
        tags=["engine-oil", "european", "acea", "specifications", "maintenance", "viscosity"],
        source="European Lubrication Standards Manual",
    ),
    KnowledgeDocument(
        id="european-electrical-systems",
        title="European Vehicle Electrical Systems",
        content="""
EUROPEAN ELECTRICAL STANDARDS:
- 12V systems standard (24V for commercial vehicles)
- Advanced CAN bus networks
- LIN bus for comfort systems
- FlexRay for safety-critical systems

BATTERY SPECIFICATIONS:
- AGM batteries for start-stop systems
- EFB (Enhanced Flooded Battery) technology
- Typical capacity: 60-90 Ah for passenger cars
- Cold cranking amps (CCA) critical in Baltic climate

START-STOP SYSTEMS:
- Standard on most European vehicles since 2010
- Requires special AGM/EFB batteries
- Battery monitoring systems integrated
- Potential fuel savings: 5-10% in city driving

EUROPEAN LIGHTING REGULATIONS:
- ECE headlight standards mandatory
- Daytime Running Lights (DRL) required since 2011
- Xenon/HID headlights with washers and leveling
- LED technology increasingly common

CHARGING SYSTEMS:
- Smart alternators with battery management
- Regenerative braking systems
- Variable voltage charging (12-15V)
- Load reduction strategies for fuel economy

ELECTRICAL PROBLEMS COMMON IN EUROPE:
1. Corrosion from road salt (winter conditions)
2. Moisture ingress in wet climates
3. Battery drain from comfort systems
4. CAN bus communication errors
5. Xenon ballast failures

EUROPEAN BRAND ELECTRICAL CHARACTERISTICS:
- BMW: Complex bus systems, coding required for components
- Mercedes: Intelligent alternator control, battery registration
- Audi/VW: Longlife batteries, sophisticated electronics
- Volvo: Safety-focused electrical systems
- PSA: Multiplexed wiring, BSI control units

DIAGNOSTIC PROCEDURES:
- OBD-II mandatory since 2001 (Euro 3)
- EOBD (European OBD) standards
- Manufacturer-specific protocols required
- CAN bus communication testing essential

BATTERY MAINTENANCE:
- AGM batteries require specific chargers
- Battery registration needed after replacement
- Temperature compensation charging
- Load testing in cold conditions

COST ESTIMATES (EUR):
- Standard battery: €80-150
- AGM/EFB battery: €120-250
- Alternator replacement: €300-600
- Starter motor: €200-500
- Xenon ballast: €150-400
- CAN bus repair: €100-500

WINTER ELECTRICAL ISSUES:
- Reduced battery capacity in cold
- Increased electrical load (heating, lights)
- Starter motor stress in cold conditions
- Potential for ice damage to wiring

SAFETY SYSTEMS:
- ABS/ESP integration with electrical systems
- Airbag systems with crash sensors
- Emergency call systems (eCall) mandatory 2018+
- Advanced driver assistance systems (ADAS)

CHARGING INFRASTRUCTURE:
- Type 2 charging standard for EVs
- 230V household charging capability
- Three-phase charging common
- Rapid charging network expanding

REGULATIONS AND STANDARDS:
- EMC (Electromagnetic Compatibility) requirements
- CE marking for aftermarket electronics
- E-mark approval for lighting components
- Specific disposal requirements for batteries
        """,
        category="Electrical",
        tags=["electrical", "european", "battery", "charging", "start-stop", "regulations"],
        source="European Electrical Systems Guide",
    ),
    KnowledgeDocument(
        id="lithuanian-technical-inspection",
        title="Lithuanian Technical Inspection (Techninė apžiūra)",
        content="""
LITHUANIAN TECHNICAL INSPECTION REQUIREMENTS:
- Mandatory for all vehicles over 4 years old
- New vehicles: First inspection after 4 years
- 4-10 years: Every 2 years
- Over 10 years: Annual inspection

INSPECTION STATIONS:
- Authorized by Regitra (Lithuanian vehicle registration authority)
- Digital inspection equipment required
- Results uploaded to central database
- Certificate valid across EU

INSPECTION CATEGORIES:
1. Category M1: Passenger cars up to 8 seats
2. Category N1: Light commercial vehicles up to 3.5t
3. Category L: Motorcycles and mopeds
4. Trailers: Separate inspection required

KEY INSPECTION POINTS:
- Braking system efficiency test
- Steering system play measurement
- Suspension condition and alignment
- Exhaust emissions testing
- Lighting system functionality
- Tire condition and tread depth

BRAKE TESTING:
- Roller brake tester required
- Minimum efficiency: 50% for service brakes
- Handbrake efficiency: 16% minimum
- Brake imbalance limits: <30%
- ABS/ESP system functionality

EMISSIONS TESTING:
- Gasoline: Lambda, CO content
- Diesel: Opacity, smoke density
- Euro 5/6: OBD emissions check
- AdBlue system verification (diesel)

SUSPENSION AND STEERING:
- Ball joint wear measurement
- Shock absorber effectiveness
- Wheel bearing condition
- Steering play limits
- Alignment within specifications

LIGHTING REQUIREMENTS:
- Headlight aim adjustment
- All bulbs functioning
- DRL (Daytime Running Lights) operational
- No illegal modifications
- Proper beam pattern

TIRE INSPECTION:
- Minimum tread depth: 1.6mm
- Winter tires during mandatory period
- No cuts or bulges in sidewalls
- Proper tire pressure
- TPMS functionality (if equipped)

COMMON FAILURE REASONS:
1. Brake efficiency below minimum
2. Excessive exhaust emissions
3. Defective lighting
4. Worn suspension components
5. Illegal modifications

COSTS (EUR):
- Standard car inspection: €18.50
- Motorcycle inspection: €13.00
- Re-inspection (partial): €9.25
- Complete re-inspection: Full price
- Additional tests: Variable pricing

PREPARATION TIPS:
- Service brakes before inspection
- Check all lights and indicators
- Ensure proper tire condition
- Fix any known mechanical issues
- Clear any stored fault codes

SEASONAL CONSIDERATIONS:
- Winter tire requirement: Nov 10 - Apr 10
- Summer tire performance in winter fails inspection
- Road salt damage inspection
- Heating system functionality check

DOCUMENTATION REQUIRED:
- Vehicle registration certificate
- Valid insurance certificate
- Previous inspection certificate
- Driver's license for identification

INSPECTION VALIDITY:
- Certificate valid for 24 months (new cars)
- Certificate valid for 12 months (cars >10 years)
- Valid across all EU countries
- Required for vehicle registration transfer

PENALTIES FOR EXPIRED INSPECTION:
- Fine: €30-60 for expired certificate
- Vehicle registration suspension possible
- Insurance may be invalidated
- Cannot legally drive on public roads
        """,
        category="Regulations",
        tags=["lithuania", "technical-inspection", "regitra", "testing", "requirements", "legal"],
        source="Regitra Technical Inspection Manual",
    ),
    # COMPREHENSIVE BASIC AUTOMOTIVE KNOWLEDGE WITH EUROPEAN CONTEXT
    KnowledgeDocument(
        id="engine-overheating-european",
        title="Engine Overheating - European Climate Considerations",
        content="""
SYMPTOMS: Temperature gauge in red zone, steam from hood, coolant warning light, loss of power.

EUROPEAN CLIMATE FACTORS:
- Baltic winters stress cooling systems with freeze/thaw cycles
- Summer temperatures across Southern Europe (40°C+) strain cooling
- Mountain driving in Alps/Pyrenees increases cooling demands
- Stop-start traffic in European cities causes overheating

COMMON CAUSES:
1. Low Coolant: Check expansion tank when cool. European cars use long-life coolants (G12+)
2. Thermostat Failure: Stuck closed, common in high-mileage European vehicles
3. Radiator Problems: Clogged with salt/debris, damaged cooling fans
4. Water Pump Failure: Belt-driven or electric, varies by European brand
5. Head Gasket: Critical on turbocharged European engines

EUROPEAN BRAND SPECIFICS:
- BMW: N20/N26 engines prone to water pump failure around 100,000 km
- Mercedes: OM642 diesel engines have coolant leak issues
- VW/Audi: 2.0 TDI thermostat housing leaks common
- Volvo: D5 engines - EGR cooler failures cause overheating
- PSA: 1.6 HDI - head gasket problems in high-mileage vehicles

IMMEDIATE ACTIONS:
1. Pull over safely, engine off immediately
2. Never open coolant cap when hot (pressurized to 1.5 bar)
3. Check coolant level in expansion tank when cool
4. Look for pink/blue coolant leaks (European G12 coolants)

DIAGNOSTIC PROCEDURES:
1. Pressure test cooling system (1.5 bar standard)
2. Check thermostat opening temperature (usually 87-92°C)
3. Test radiator cap pressure relief
4. Inspect water pump for leaks/play
5. Compression test for head gasket integrity

REPAIR COSTS (EUR):
- Thermostat replacement: €120-250
- Radiator repair/replacement: €300-800
- Water pump replacement: €400-900
- Head gasket repair: €1,200-2,500
- Coolant system flush: €80-150

PREVENTION:
- Use OEM-specification long-life coolant (VW G12+, BMW Blue, etc.)
- Check coolant level monthly
- Replace thermostat every 100,000 km
- Flush cooling system every 4-5 years or per manufacturer schedule

WINTER CONSIDERATIONS:
- Antifreeze protection to -35°C in Baltic states
- Block heater use in extreme cold
- Check coolant strength before winter season
        """,
        category="Cooling",
        tags=["overheating", "coolant", "radiator", "thermostat", "european", "climate"],
        source="European Cooling System Guide",
    ),
    KnowledgeDocument(
        id="transmission-problems-european",
        title="Transmission Issues - European Vehicles",
        content="""
EUROPEAN TRANSMISSION TYPES:
- Manual: 6-speed common, sporty driving style
- Automatic: DSG/S-Tronic (VAG), 7G-Tronic (Mercedes), ZF 8HP
- CVT: Limited use, mainly Nissan and some Toyota models
- Dual-clutch: Popular in European performance cars

AUTOMATIC TRANSMISSION SYMPTOMS:
- Harsh shifting: Common in DSG/PDK systems
- Mechatronic failures: VW/Audi DSG specific issue
- Torque converter problems: Traditional automatics
- Software issues: Require manufacturer-specific updates

MANUAL TRANSMISSION ISSUES:
- Clutch wear: Typical lifespan 120,000-200,000 km European driving
- Gearbox oil: Often lifetime fill, but change recommended at 100,000 km
- Synchronizer wear: Hard shifting into specific gears
- Clutch hydraulics: Master/slave cylinder failures

EUROPEAN BRAND SPECIFICS:
- VW/Audi DSG: Mechatronic unit failures, software updates critical
- BMW: ZF automatics generally reliable, manual clutch wear common
- Mercedes: 7G-Tronic valve body issues, NAG1 transmission problems
- Volvo: Powershift (Ford) transmission problems in some models
- PSA: EGS6 transmission software issues

DIAGNOSTIC STEPS:
1. Scan for transmission fault codes (manufacturer-specific)
2. Check transmission fluid level/condition (where accessible)
3. Road test for shift quality and timing
4. Adaptation reset after repairs (VAG vehicles especially)
5. Software version check for known updates

CLUTCH REPLACEMENT INDICATORS (MANUAL):
- Slipping: Engine revs without acceleration
- High bite point: Clutch engages near top of pedal travel
- Difficulty shifting: Worn clutch not fully disengaging
- Juddering: Worn flywheel or pressure plate

MAINTENANCE:
- Manual: Gearbox oil change every 60,000-100,000 km
- Automatic/DSG: Service every 40,000-80,000 km (despite "lifetime" claims)
- Clutch adjustment: Check every 20,000 km
- Software updates: Important for dual-clutch systems

REPAIR COSTS (EUR):
- Manual transmission oil change: €100-200
- Automatic/DSG service: €200-400
- Clutch replacement: €800-1,800
- DSG mechatronic unit: €2,000-4,000
- Automatic transmission rebuild: €2,500-5,000

DRIVING TIPS FOR LONGEVITY:
- Manual: Proper clutch technique, don't ride clutch
- Automatic: Allow warmup in cold weather, use correct ATF
- DSG: Smooth acceleration, regular service intervals critical
- All types: Avoid aggressive driving until warmed up

WINTER CONSIDERATIONS:
- Thicker transmission fluids in cold affect shifting
- Clutch hydraulics can freeze in extreme cold
- Allow extended warmup for smooth operation
        """,
        category="Transmission",
        tags=["transmission", "clutch", "dsg", "automatic", "manual", "european"],
        source="European Transmission Service Guide",
    ),
    KnowledgeDocument(
        id="air-conditioning-heating-european",
        title="Air Conditioning and Heating - European Climate Systems",
        content="""
EUROPEAN CLIMATE CONTROL FEATURES:
- Automatic climate control standard on most vehicles
- Heat pump systems in electric vehicles
- Auxiliary heating (Webasto/Eberspächer) common in cold climates
- Pollen filters mandatory, important for allergy season

AC NOT COOLING:
- R134a refrigerant being phased out for R1234yf in new vehicles
- System pressures different for European specifications
- Compressor clutch issues in stop-start vehicles
- Evaporator drainage problems in humid climates

HEATING PROBLEMS:
- Insufficient heat: Common in efficient European engines
- Auxiliary heater malfunctions: Diesel/petrol fired heaters
- Blend door problems: Complex climate control systems
- Coolant circulation: Low coolant affects cabin heating

EUROPEAN BRAND SPECIFICS:
- BMW: Automatic climate control programming complex
- Mercedes: Thermatic system issues, refrigerant leak detection
- VW/Audi: Climatronic system, common actuator failures
- Volvo: PTC auxiliary heaters, climate sensor issues
- PSA: Automatic climate control, blend door motor failures

REFRIGERANT SPECIFICATIONS:
- R134a: Older vehicles (pre-2017)
- R1234yf: New vehicles (2017+), more expensive
- System capacity: Typically 450-650g for passenger cars
- PAG oil: Specific to refrigerant type, not interchangeable

MAINTENANCE SCHEDULE:
- AC performance check: Annually before summer
- Pollen filter replacement: Every 15,000-20,000 km or annually
- Refrigerant leak test: Every 2-3 years
- System evacuation/recharge: Every 4-5 years
- Auxiliary heater service: Annually before winter

COMMON PROBLEMS:
1. Compressor failure: Often due to lack of lubrication
2. Evaporator leaks: Corrosion in humid climates
3. Condenser damage: Stone chips, salt corrosion
4. Electrical issues: Climate control modules, sensors
5. Auxiliary heater problems: Fuel supply, glow plugs

SEASONAL MAINTENANCE:
- Spring: Test AC before hot weather, replace pollen filter
- Summer: Check for adequate cooling, clean condenser
- Fall: Test heating system, check auxiliary heater
- Winter: Use defrost regularly, check coolant level

REPAIR COSTS (EUR):
- AC recharge (R134a): €80-120
- AC recharge (R1234yf): €150-250
- Compressor replacement: €600-1,200
- Evaporator replacement: €800-1,500
- Auxiliary heater repair: €300-800
- Pollen filter replacement: €25-60

ENERGY EFFICIENCY:
- Use recirculation mode for faster cooling
- Pre-cooling with remote start (if available)
- Parking in shade reduces AC load
- Regular maintenance improves efficiency

LEGAL REQUIREMENTS:
- R1234yf mandatory on new vehicles (EU regulation)
- Proper refrigerant recovery during service
- Technician certification required for refrigerant work
- Environmental disposal of old refrigerants
        """,
        category="Climate Control",
        tags=["ac", "heating", "climate", "refrigerant", "auxiliary-heater", "european"],
        source="European HVAC Systems Manual",
    ),
    KnowledgeDocument(
        id="oil-change-european-standards",
        title="Oil Changes and Engine Maintenance - European Standards",
        content="""
EUROPEAN ENGINE OIL REQUIREMENTS:
- ACEA specifications mandatory for warranty compliance
- Manufacturer-specific approvals critical (VW 504.00, BMW LL-04, etc.)
- Long-life intervals: 15,000-30,000 km with appropriate oils
- Climate considerations: 0W grades essential for Baltic winters

OIL CHANGE INTERVALS BY TYPE:
- Conventional oil: 10,000-15,000 km (basic European cars)
- Synthetic blend: 15,000-20,000 km
- Full synthetic long-life: 20,000-30,000 km
- Severe conditions: Reduce intervals by 30-50%

CHECKING OIL LEVEL:
- Electronic dipsticks common on modern European cars
- Check when engine warm but off for 5-10 minutes
- iDrive/MMI systems show oil level on BMW/Audi
- Oil level sensors can fail, causing false warnings

EUROPEAN BRAND OIL REQUIREMENTS:
- BMW: LL-01 (gasoline), LL-04 (diesel), specific viscosity grades
- Mercedes: MB Approval 229.31/229.51/229.52
- VW/Audi: VW 504.00/507.00 for modern TDI engines
- Volvo: VCC RBS0-2AE for Drive-E engines
- PSA: PSA B71 2290/2300 for modern engines

OIL CONDITION INDICATORS:
- Service indicator systems: Follow manufacturer reset procedures
- Oil life monitoring: Considers driving conditions, not just mileage
- Visual inspection: Fresh oil amber, avoid black or milky oil
- Oil analysis: Available at some service centers for fleet vehicles

FILTER REPLACEMENT:
- Paper filters: Standard, adequate for normal service
- Synthetic media: Better filtration, longer service life
- Bypass filters: Some diesel engines have additional filters
- Change with every oil service, use OEM or equivalent quality

OTHER FLUIDS TO CHECK:
- AdBlue (DEF): Diesel exhaust fluid, consumption 3-5% of fuel
- Brake fluid: DOT 4 standard, hygroscopic, change every 2-3 years
- Power steering: Many European cars now electric assisted
- Coolant: G12+ long-life, check concentration before winter

AIR FILTER MAINTENANCE:
- Paper filters: Replace every 20,000-30,000 km
- Performance filters: K&N style, cleanable but affect warranty
- Cabin filters: Pollen/carbon, replace annually or 15,000 km
- Diesel particulate filters: Separate system, regeneration required

SPARK PLUG MAINTENANCE (GASOLINE):
- Iridium plugs: 60,000-100,000 km intervals
- Platinum plugs: 40,000-80,000 km intervals
- Copper/nickel: 20,000-40,000 km (older vehicles)
- Turbo engines: May require shorter intervals due to heat

PREVENTIVE MAINTENANCE SCHEDULE:
- Oil service: 10,000-30,000 km (depending on oil type)
- Major service: Every 20,000-40,000 km
- Timing belt: 80,000-160,000 km (interference engines critical)
- Spark plugs: 40,000-100,000 km
- Air filter: 20,000-30,000 km

COST BREAKDOWN (EUR):
- Basic oil change: €60-120
- Premium long-life oil service: €120-200
- Major service: €200-400
- Timing belt replacement: €400-800
- Spark plug replacement: €100-250

SIGNS OF ENGINE PROBLEMS:
- Oil consumption: More than 1L per 1000km needs investigation
- Metal particles: Bearing wear, internal damage
- Coolant in oil: Head gasket or heat exchanger problems
- Performance issues: May indicate carbon buildup, common in GDI engines

ENVIRONMENTAL CONSIDERATIONS:
- Used oil recycling mandatory in EU
- Proper filter disposal required
- Extended intervals reduce environmental impact
- Synthetic oils generally more environmentally friendly
        """,
        category="Engine",
        tags=["oil", "maintenance", "acea", "long-life", "european", "service"],
        source="European Engine Maintenance Standards",
    ),
    KnowledgeDocument(
        id="common-european-car-problems",
        title="Most Common European Car Problems by Brand",
        content="""
BMW COMMON ISSUES:
- N47/N57 Diesel: Timing chain stretch, swirl flap failures
- N20/N26 Gasoline: Timing chain, water pump, valve cover leaks
- Electrical: Window regulators, iDrive system glitches
- Suspension: Control arm bushings, strut mounts
- Cooling: Water pump, thermostat housing, radiator leaks

MERCEDES-BENZ COMMON ISSUES:
- OM642 Diesel: Glow plug failures, oil cooler leaks, DPF problems
- M272/M273 Gasoline: Balance shaft wear, intake manifold leaks
- Electrical: SAM modules, central locking issues
- Air suspension: Compressor failures, air leaks (S-Class, E-Class)
- Transmission: 7G-Tronic valve body issues

VW/AUDI COMMON ISSUES:
- 2.0 TDI: Timing belt, oil pump, EGR valve problems
- 1.8/2.0 TSI: Carbon buildup, timing chain, water pump
- DSG Transmission: Mechatronic unit, clutch wear
- Electrical: Central convenience modules, MMI system issues
- Suspension: Control arms, shock absorbers

VOLVO COMMON ISSUES:
- D5 Diesel: Swirl flaps, EGR valve, turbocharger issues
- T5/T6 Gasoline: PCV system, angle gear (AWD), transmission issues
- Electrical: CEM (Central Electronic Module), tailgate issues
- Suspension: Four-C system problems, strut mounts
- Climate: Auxiliary heater problems in cold climates

PEUGEOT/CITROËN COMMON ISSUES:
- 1.6 HDI: Timing belt, turbocharger, EGR valve problems
- 1.6 THP: Timing chain, carbon buildup, water pump
- Electrical: BSI (Built-in Systems Interface) failures
- Suspension: Hydropneumatic system (older models), shock absorbers
- Climate: Blend door actuators, air conditioning compressor

COST ESTIMATES FOR COMMON REPAIRS (EUR):
- Timing chain replacement: €800-1,500
- Turbocharger replacement: €1,200-2,500
- DSG service/repair: €300-3,000
- Air suspension compressor: €400-800
- EGR valve replacement: €300-600
- Water pump replacement: €300-700

PREVENTIVE MEASURES:
- Regular oil changes with correct specification
- Quality fuel and additives for DPF systems
- Software updates for known issues
- Proper driving style (especially diesel vehicles)
- Regular diagnostic scans for early problem detection

HIGH-MILEAGE CONSIDERATIONS:
- 100,000+ km: Timing belts/chains, water pumps, suspension components
- 150,000+ km: Turbocharger, transmission service, major electrical issues
- 200,000+ km: Engine rebuilds, major suspension overhauls
- Annual inspections become more critical

SEASONAL PREPARATION:
- Winter: Battery test, coolant check, tire changes
- Summer: AC service, cooling system check, brake inspection
- Year-round: Regular oil changes, software updates

WARRANTY CONSIDERATIONS:
- Use OEM or OE-equivalent parts when possible
- Maintain service records for warranty claims
- Software updates may void aftermarket modifications
- Extended warranties available for high-maintenance vehicles

DIAGNOSTIC TOOLS:
- OBD-II scanners: Basic fault code reading
- Manufacturer-specific: VCDS (VAG), INPA (BMW), STAR (Mercedes)
- Professional: Snap-on, Bosch, Launch for comprehensive diagnosis
        """,
        category="General",
        tags=["common-problems", "bmw", "mercedes", "vw", "audi", "volvo", "peugeot", "european"],
        source="European Vehicle Reliability Database",
    ),
]


def get_knowledge_documents() -> List[KnowledgeDocument]:
    """Get all comprehensive European knowledge base documents."""
    return EUROPEAN_KNOWLEDGE_BASE


def get_documents_by_category(category: str) -> List[KnowledgeDocument]:
    """Get documents filtered by category."""
    return [doc for doc in EUROPEAN_KNOWLEDGE_BASE if doc.category.lower() == category.lower()]


def get_documents_by_tags(tags: List[str]) -> List[KnowledgeDocument]:
    """Get documents that have any of the specified tags."""
    return [
        doc
        for doc in EUROPEAN_KNOWLEDGE_BASE
        if any(tag.lower() in [t.lower() for t in doc.tags] for tag in tags)
    ]


def search_documents(query: str) -> List[KnowledgeDocument]:
    """Simple text search in document content."""
    query_lower = query.lower()
    results = []

    for doc in EUROPEAN_KNOWLEDGE_BASE:
        if (
            query_lower in doc.title.lower()
            or query_lower in doc.content.lower()
            or any(query_lower in tag.lower() for tag in doc.tags)
        ):
            results.append(doc)

    return results


def get_country_specific_documents(country_code: str) -> List[KnowledgeDocument]:
    """Get documents specific to a country."""
    country_specific = {
        "LT": ["lithuanian-technical-inspection", "european-tire-regulations"],
        "DE": [
            "european-brake-systems",
            "european-electrical-systems",
            "european-tire-regulations",
        ],
        "GB": ["european-tire-regulations", "european-brake-systems"],
        "LV": ["european-tire-regulations", "engine-overheating-european"],
        "EE": ["european-tire-regulations", "engine-overheating-european"],
        "SE": ["european-tire-regulations", "european-electrical-systems"],
        "NO": ["european-tire-regulations", "european-electrical-systems"],
        "FI": ["european-tire-regulations", "european-electrical-systems"],
        # Add more country-specific mappings as needed
    }

    if country_code not in country_specific:
        return []

    relevant_ids = country_specific[country_code]
    return [doc for doc in EUROPEAN_KNOWLEDGE_BASE if doc.id in relevant_ids]


# Legacy compatibility function
def get_european_knowledge_documents() -> List[KnowledgeDocument]:
    """Legacy function - use get_knowledge_documents() instead."""
    return get_knowledge_documents()
