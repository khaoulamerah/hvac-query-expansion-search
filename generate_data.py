import random

import pandas as pd

random.seed(42)
BRANDS = ["AquaPro", "ThermoMax", "HydroLine", "PipeMaster", "FlowTech", "HeatWave", "EcoTherm", "ClearFlow"]

# (name, category, description with {v}, base price, variants)
F = [
 ("Combi Boiler", "Boilers", "Wall-mounted gas combi boiler {v} for central heating and instant hot water, high efficiency condensing", 900, ["24kW", "30kW", "35kW"]),
 ("System Boiler", "Boilers", "Gas system boiler {v} for central heating with hot water cylinder, sealed system", 1000, ["18kW", "24kW", "30kW"]),
 ("Electric Boiler", "Boilers", "Electric boiler {v} for central heating, no flue needed, compact", 1100, ["6kW", "9kW", "12kW"]),
 ("Storage Water Heater", "Water Heating", "Electric storage tank water heater {v} with thermostat for hot water supply", 250, ["50L", "80L", "100L", "150L"]),
 ("Hot Water Cylinder", "Water Heating", "Unvented stainless steel hot water cylinder {v} with immersion heater", 400, ["120L", "180L", "250L"]),
 ("Panel Radiator", "Radiators", "Double panel steel radiator {v} for room heating with wall brackets", 50, ["600x600", "600x1000", "600x1400", "300x1200"]),
 ("Towel Rail Radiator", "Radiators", "Heated bathroom towel rail radiator {v}, chrome or white finish", 70, ["500x800", "500x1200", "600x1600"]),
 ("Thermostatic Radiator Valve", "Valves", "TRV {v} pattern for controlling radiator temperature, chrome", 15, ["angled", "straight", "corner"]),
 ("Radiator Bleed Key", "Accessories", "Key to bleed trapped air from radiators and remove cold spots", 3, [""]),
 ("Smart Thermostat", "Controls", "{v} programmable room thermostat controller for heating system with app control", 60, ["WiFi", "Wireless", "Wired"]),
 ("Underfloor Thermostat", "Controls", "Digital floor thermostat with sensor for underfloor heating {v}", 45, ["touchscreen", "dial"]),
 ("Circulator Pump", "Pumps", "Variable speed central heating circulation pump {v}", 110, ["15-50", "25-60", "25-80"]),
 ("Underfloor Heating Kit", "Underfloor Heating", "Electric underfloor heating mat {v} for bathroom, kitchen or tiled floors", 120, ["5m2", "10m2", "15m2", "20m2"]),
 ("Expansion Vessel", "Heating Parts", "Pressure expansion vessel {v} for sealed heating systems", 30, ["8L", "12L", "18L", "24L"]),
 ("Magnetic System Filter", "Heating Parts", "Magnetic filter {v} removes sludge and debris from heating water, protects boiler", 70, ["22mm", "28mm"]),
 ("Boiler Flue Extension", "Heating Parts", "Boiler flue pipe extension kit {v} for gas boilers", 25, ["500mm", "1m"]),
 ("Heating System Inhibitor", "Consumables", "Central heating inhibitor liquid {v} prevents corrosion and limescale", 14, ["500ml", "1L"]),
 ("Radiator Cleaner", "Consumables", "Heating system cleaner descaler {v} removes sludge and restores heat output", 16, ["1L", "5L"]),
 ("Copper Pipe", "Pipes", "Copper tube {v} for water supply and heating pipework", 14, ["15mm 3m", "22mm 3m", "28mm 3m", "15mm coil 10m"]),
 ("PEX Pipe", "Pipes", "Flexible plastic piping {v} for hot and cold water plumbing and underfloor heating", 40, ["16mm 25m", "16mm 50m", "20mm 50m"]),
 ("PVC Waste Pipe", "Pipes", "Waste drain pipe {v} for sinks, basins and soil", 8, ["32mm", "40mm", "50mm", "110mm soil"]),
 ("Compression Elbow", "Fittings", "Brass compression elbow fitting {v} joins copper pipes without soldering", 2, ["15mm", "22mm", "28mm"]),
 ("Compression Tee", "Fittings", "Brass equal tee fitting {v} branches copper pipes", 3, ["15mm", "22mm", "28mm"]),
 ("Pipe Coupling", "Fittings", "Straight coupler {v} connecting two pipes", 2, ["15mm", "22mm", "28mm"]),
 ("Ball Valve", "Valves", "Quarter turn brass shutoff valve stopcock {v} for water isolation", 6, ["1/2 inch", "3/4 inch", "1 inch"]),
 ("Isolation Valve", "Valves", "Service isolating valve {v} for taps, appliances and radiators", 4, ["15mm", "22mm"]),
 ("Pressure Reducing Valve", "Valves", "Water pressure reducing valve {v} protects pipes and taps from high pressure", 35, ["15mm", "22mm"]),
 ("Kitchen Mixer Tap", "Taps", "Single lever kitchen faucet mixer {v} with swivel spout", 55, ["chrome", "brushed steel", "black"]),
 ("Basin Tap", "Taps", "Bathroom basin faucet {v} with ceramic cartridge", 40, ["chrome", "waterfall", "monobloc"]),
 ("Bath Shower Mixer Tap", "Taps", "Bath filler tap with shower handset {v}", 90, ["chrome", "black"]),
 ("Tap Cartridge", "Spare Parts", "Replacement ceramic cartridge {v} to fix dripping leaking taps", 9, ["35mm", "40mm"]),
 ("Flexible Tap Connector", "Spare Parts", "Flexible hose tap connector {v} for taps and basins", 3, ["15mm 300mm", "15mm 500mm"]),
 ("Thermostatic Shower Mixer", "Showers", "Shower valve {v} with temperature control and rain head", 100, ["exposed", "concealed", "bar"]),
 ("Electric Shower", "Showers", "Instantaneous electric shower {v} heats water on demand", 90, ["8.5kW", "9.5kW", "10.5kW"]),
 ("Close Coupled Toilet", "Sanitary", "Ceramic WC toilet {v} with cistern and seat", 110, ["soft close", "rimless", "compact"]),
 ("Wall Hung Toilet", "Sanitary", "Wall hung WC pan {v} for modern bathroom, needs concealed frame", 150, ["rimless", "standard"]),
 ("Wash Basin", "Sanitary", "Ceramic sink basin {v} for bathroom", 50, ["pedestal", "wall hung", "countertop"]),
 ("Kitchen Sink", "Sanitary", "Stainless steel kitchen sink {v} with waste", 80, ["single bowl", "1.5 bowl", "double bowl"]),
 ("Water Softener", "Water Treatment", "Water softener {v} reduces limescale in hard water areas, protects boiler and taps", 350, ["compact", "family"]),
 ("PTFE Thread Seal Tape", "Consumables", "Tape to seal threaded joints and stop leaks", 2, [""]),
 ("Pipe Repair Clamp", "Consumables", "Emergency clamp {v} to stop pipe leak and burst repair", 10, ["15mm", "22mm"]),
 ("Pipe Insulation Foam", "Consumables", "Foam lagging {v} to prevent frozen pipes and heat loss", 3, ["15mm 2m", "22mm 2m"]),
 ("Silicone Sealant", "Consumables", "Waterproof sealant {v} for sinks, baths and showers", 5, ["white", "clear"]),
 ("Drain Unblocker", "Consumables", "Drain cleaner {v} clears blocked sinks, basins and pipes", 6, ["1L", "2L"]),
 ("Water Leak Detector", "Controls", "Smart water leak sensor {v} alerts you to leaks under sinks and boilers", 25, ["WiFi", "battery"]),
]

rows = []
for name, cat, desc, base, variants in F:
    for i, v in enumerate(variants):
        full = f"{name} {v}".strip()
        price = round(base * (1 + 0.22 * i) * random.uniform(0.95, 1.1), 2)
        rows.append({"name": full, "category": cat, "brand": random.choice(BRANDS),
                     "description": " ".join(desc.format(v=v).split()), "price": price})
df = pd.DataFrame(rows)
df.insert(0, "id", range(1, len(df) + 1))
df.to_csv("data/products.csv", index=False)
print(len(df), "products saved to data/products.csv")