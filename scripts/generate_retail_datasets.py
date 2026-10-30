#!/usr/bin/env python3
"""generate_retail_datasets.py

Deterministic generator for Retail Product Catalog Enrichment SKU datasets
(60, 100, and 250 SKUs) and pre-computed Stage 1 dgem evaluation receipts
across all 3 retail policy templates:
  1. retail_merchandising_compliance (4 decision slots per SKU)
  2. retail_deep_taxonomy            (4 decision slots per SKU, including 32-option tournament category)
  3. retail_catalog_qa               (3 decision slots per SKU)

Each dataset contains ~75% clear Stage 1 SKUs and ~25% ambiguous edge-case SKUs
(cross-category hybrids, borderline regulatory/wellness claims, subtle title-vs-spec
contradictions, and missing critical attributes) designed to trigger Stage 2
escalation to `gemini-3.5-flash-lite` when normalized Shannon entropy (Hesitation %)
exceeds the calibrated threshold (tau = 0.16).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

# 32 Fine-Grained Google Product Taxonomy Categories (cat_01 .. cat_32)
# Evaluates via dgem's 2-Stage Tournament Bracket Mode (>26 options).
RETAIL_TAXONOMY_32: Dict[str, str] = {
    "cat_01": "Apparel & Accessories > Clothing > Outerwear > Coats & Jackets: insulated parkas, rain shells, down jackets, tailored blazers",
    "cat_02": "Apparel & Accessories > Clothing > Activewear & Athletic Apparel: compression leggings, running shorts, moisture-wicking tops",
    "cat_03": "Apparel & Accessories > Clothing > Everyday Tops, Dresses & Knitwear: merino sweaters, button-down shirts, casual dresses",
    "cat_04": "Apparel & Accessories > Shoes > Athletic & Trail Running Footwear: marathon road shoes, trail runners, cross-training sneakers",
    "cat_05": "Apparel & Accessories > Shoes > Boots, Dress & Casual Footwear: Goodyear-welt leather boots, loafers, waterproof chelsea boots",
    "cat_06": "Apparel & Accessories > Handbags, Wallets, Watches & Eyewear: polarized sunglasses, leather totes, RFID wallets, field watches",
    "cat_07": "Electronics > Computers > Laptops, Ultrabooks & Mobile Workstations: notebook computers, 2-in-1 convertibles, creator laptops",
    "cat_08": "Electronics > Computers > Monitors, Mechanical Keyboards & Docks: 4K displays, ergonomic split keyboards, Thunderbolt docks",
    "cat_09": "Electronics > Audio > Headphones, Wireless Earbuds & Studio Speakers: ANC headphones, bookshelf speakers, Bluetooth soundbars",
    "cat_10": "Electronics > Communications > Smartphones, Cases & MagSafe Chargers: unlocked 5G handsets, GaN chargers, rugged phone cases",
    "cat_11": "Electronics > Smart Home > Security Cameras, Thermostats & Lighting: Matter smart bulbs, PoE doorbells, smart HVAC sensors",
    "cat_12": "Electronics > Video Game Consoles, VR Headsets & Pro Controllers: handheld gaming PCs, Hall-effect gamepads, sim racing wheels",
    "cat_13": "Electronics > Cameras, Gimbal Stabilizers & Action Video Gear: mirrorless camera bodies, 3-axis gimbals, waterproof action cams",
    "cat_14": "Home & Garden > Kitchen & Dining > Countertop Espresso & Small Appliances: dual-boiler espresso machines, air fryers, high-speed blenders",
    "cat_15": "Home & Garden > Kitchen & Dining > Cookware, Dutch Ovens & Chef Cutlery: tri-ply stainless skillets, enameled cast iron, forged knives",
    "cat_16": "Home & Garden > Kitchen & Dining > Tableware, Vacuum Drinkware & Barware: stoneware dinner sets, insulated tumblers, crystal glassware",
    "cat_17": "Home & Garden > Furniture > Ergonomic Office, Living & Bedroom Furniture: mesh task chairs, standing desks, solid oak bed frames",
    "cat_18": "Home & Garden > Linens & Bedding > Mattresses, Duvets & Cooling Pillows: percale sheet sets, down comforters, memory foam pillows",
    "cat_19": "Home & Garden > Household Appliances > HEPA Purifiers, Vacuums & Dehumidifiers: LiDAR robot vacuums, true-HEPA air purifiers, humidifiers",
    "cat_20": "Home & Garden > Lawn, Patio, Pellet Grills & Outdoor Living: Wi-Fi pellet smokers, modular patio sofas, cordless string trimmers",
    "cat_21": "Health & Beauty > Personal Care > Facial Skincare, Retinol Serums & SPF: vitamin C serums, mineral sunscreens, barrier moisturizers",
    "cat_22": "Health & Beauty > Personal Care > Cosmetics, Lip Oils & Mineral Makeup: liquid foundations, eyeshadow palettes, tinted lip balms",
    "cat_23": "Health & Beauty > Personal Care > Hair Care, Ionic Dryers & Styling Tools: brushless hair dryers, ceramic flat irons, bond-repair masks",
    "cat_24": "Health & Beauty > Personal Care > Fine Fragrance, Bath & Body Grooming: eau de parfum, botanical body washes, safety razors",
    "cat_25": "Health & Beauty > Health Care > Percussion Massagers & Recovery Devices: deep-tissue massage guns, red-light therapy panels, tens units",
    "cat_26": "Sporting Goods > Outdoor Recreation > Backpacking Tents, Sleeping Bags & Packs: ultralight 3-season tents, down quilts, hydration packs",
    "cat_27": "Sporting Goods > Exercise & Fitness > Adjustable Dumbbells, Rowing & Yoga: cast-iron kettlebells, folding rowers, natural rubber yoga mats",
    "cat_28": "Sporting Goods > Cycling, Paddleboarding & Winter Snow Sports Gear: MIPS road helmets, inflatable SUP boards, ski goggles",
    "cat_29": "Food, Beverages & Tobacco > Beverages > Specialty Whole-Bean Coffee & Matcha: single-origin espresso beans, ceremonial matcha, functional teas",
    "cat_30": "Food, Beverages & Tobacco > Food Items > Artisanal Olive Oil, Honey & Pantry: cold-pressed extra virgin olive oil, raw manuka honey, dark chocolate",
    "cat_31": "Animals & Pet Supplies > Pet Nutrition > Freeze-Dried Raw Food & Joint Chews: grain-free salmon kibble, probiotic chews, freeze-dried liver treats",
    "cat_32": "Animals & Pet Supplies > Pet Gear > Orthopedic Pet Beds, Harnesses & Smart Feeders: memory-foam dog beds, no-pull reflective harnesses, automatic feeders",
}

# Clear SKU archetypes spanning all 32 categories
CLEAR_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "cat": "cat_01", "dept": "A", "season": "D", "mat": "B", "price_tier": "C",
        "brand": "AlpineCrest", "base_title": "3-Layer Gore-Tex Storm Shell Parka", "base_price": 289.0,
        "specs": ["100% recycled 70D ripstop nylon shell", "28,000mm waterproof / breathable membrane", "Helmet-compatible storm hood and pit zips", "Sizes XS-XXL with garment chest measurements"],
        "visual": "Studio front/back photo of slate-blue technical waterproof parka with taped seams.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_02", "dept": "A", "season": "B", "mat": "B", "price_tier": "B",
        "brand": "VeloceFit", "base_title": "High-Rise 7/8 Compression Running Tight", "base_price": 68.0,
        "specs": ["78% recycled polyester / 22% elastane interlock", "UPF 50+ sun protection and reflective ankle strips", "Dual side drop-in phone pockets", "Inseam: 25 inches; machine wash cold"],
        "visual": "Full-length activewear leggings in deep espresso on athletic mannequin.",
        "compliant": "yes", "completeness": "5", "gift": "no", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_03", "dept": "A", "season": "C", "mat": "A", "price_tier": "B",
        "brand": "LoomHarbor", "base_title": "100% Australian Merino Wool Crewneck Sweater", "base_price": 115.0,
        "specs": ["19.5-micron superfine mulesing-free Merino wool", "Ribbed collar, cuffs, and hem", "Naturally odor-resistant midweight 12-gauge knit", "Care: hand wash cold, lay flat to dry"],
        "visual": "Folded heather-oatmeal merino crewneck sweater showing fine rib texture.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_04", "dept": "A", "season": "B", "mat": "B", "price_tier": "C",
        "brand": "StridePeak", "base_title": "Carbon-Plate Marathon Road Running Shoe", "base_price": 195.0,
        "specs": ["Dual-density PEBA supercritical foam midsole", "Full-length carbon fiber propulsion plate", "Engineered breathable monofilament mesh upper", "Weight: 7.4 oz (Men's 9); 8mm heel-to-toe drop"],
        "visual": "Side profile of neon-coral and white road racing shoe with thick sculpted midsole.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_05", "dept": "A", "season": "C", "mat": "A", "price_tier": "C",
        "brand": "CobblerWorks", "base_title": "Goodyear-Welted Full-Grain Leather Chelsea Boot", "base_price": 245.0,
        "specs": ["Horween Chromexcel full-grain cowhide upper", "360-degree Goodyear storm welt construction", "Studded Dainite rubber outsole and cork footbed", "Available in US sizes 7-13 (D & EE widths)"],
        "visual": "Pair of dark brown leather chelsea boots with elastic side gores and pull tabs.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_06", "dept": "A", "season": "D", "mat": "C", "price_tier": "C",
        "brand": "ChronoVanguard", "base_title": "Titanium Automatic Field Watch 38mm (100m WR)", "base_price": 320.0,
        "specs": ["Grade 2 brushed titanium case (38mm diameter, 10.5mm thick)", "Sapphire crystal with inner anti-reflective coating", "24-jewel Japanese automatic mechanical movement (41-hr reserve)", "Screw-down crown, 10 ATM / 100m water resistance"],
        "visual": "Close-up of matte titanium field watch with black dial, Super-LumiNova numerals, and olive NATO strap.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_07", "dept": "B", "season": "C", "mat": "C", "price_tier": "D",
        "brand": "NovaCompute", "base_title": "ProBook 14-Inch OLED Creator Laptop (32GB RAM, 1TB NVMe)", "base_price": 1399.0,
        "specs": ["14.0-inch 2880x1800 120Hz HDR OLED display (100% DCI-P3)", "CNC-machined recycled aluminum unibody chassis (3.1 lbs)", "32GB LPDDR5X memory, 1TB PCIe Gen4 SSD", "Dual Thunderbolt 4 ports, Wi-Fi 7, 75Wh battery"],
        "visual": "Open space-gray aluminum laptop showing color-grading timeline on OLED screen.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_08", "dept": "B", "season": "C", "mat": "C", "price_tier": "C",
        "brand": "KeyForge", "base_title": "Wireless Gasket-Mount Mechanical Keyboard (75% Hot-Swap)", "base_price": 169.0,
        "specs": ["CNC 6063 anodized aluminum top and bottom case", "Hot-swappable 5-pin PCB with pre-lubed tactile switches", "Tri-mode connectivity: 2.4GHz dongle, Bluetooth 5.2, USB-C", "Double-shot PBT keycaps and QMK/VIA programmable firmware"],
        "visual": "Top-down angle of compact 75% mechanical keyboard with navy/cream PBT keycaps and brass knob.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_09", "dept": "B", "season": "D", "mat": "B", "price_tier": "C",
        "brand": "AetherAcoustic", "base_title": "Adaptive Hybrid ANC Over-Ear Wireless Headphones", "base_price": 279.0,
        "specs": ["40mm biocellulose dynamic drivers with LDAC & aptX Lossless", "Up to -45dB adaptive active noise cancellation with 6 mics", "50-hour battery life (ANC on) and USB-C fast charge", "Memory-foam protein leather earcups and folding travel case"],
        "visual": "Matte black over-ear wireless headphones alongside zippered hardshell travel case.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_10", "dept": "B", "season": "A", "mat": "B", "price_tier": "B",
        "brand": "VoltGrid", "base_title": "140W 4-Port GaN USB-C PD 3.1 Desktop & Travel Charger", "base_price": 64.0,
        "specs": ["Gallium Nitride (GaN III) architecture with foldable US prongs", "3x USB-C PD 3.1 ports (up to 140W single-port) + 1x USB-A", "Active thermal guard monitoring 100 times per second", "UL 62368-1 certified; includes 6.6ft 240W braided USB-C cable"],
        "visual": "Compact charcoal GaN wall charger next to braided USB-C cable showing port wattage labels.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_11", "dept": "B", "season": "A", "mat": "B", "price_tier": "B",
        "brand": "DomusSense", "base_title": "4K PoE Outdoor Turret Security Camera with Color Night Vision", "base_price": 129.0,
        "specs": ["1/1.8-inch 8MP CMOS sensor (3840x2160 @ 30fps)", "IP67 weatherproof metal-polymer housing (-30C to 60C)", "802.3af Power-over-Ethernet (PoE) and local microSD slot up to 512GB", "On-device person, vehicle, and package detection with RTSP/ONVIF"],
        "visual": "White weatherproof turret camera mounted under eave with warm LED spotlight.",
        "compliant": "yes", "completeness": "5", "gift": "no", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_12", "dept": "B", "season": "D", "mat": "B", "price_tier": "B",
        "brand": "HyperPad", "base_title": "Pro Hall-Effect Wireless Gaming Controller with Charging Dock", "base_price": 79.0,
        "specs": ["Drift-free magnetic Hall-effect joysticks and analog triggers", "1000Hz ultra-low-latency 2.4GHz wireless polling rate", "4 programmable rear micro-switch paddles", "Compatible with PC, Switch, Android, and SteamOS"],
        "visual": "Translucent smoke-black gamepad resting on magnetic RGB charging dock.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_13", "dept": "B", "season": "B", "mat": "F", "price_tier": "C",
        "brand": "AeroVista", "base_title": "4K/120fps Waterproof Action Camera with 1/1.3-Inch Sensor", "base_price": 349.0,
        "specs": ["1/1.3-inch CMOS sensor with 10-bit D-Log M color profile", "Waterproof to 18 meters without external housing", "360-degree HorizonSteady electronic stabilization", "Dual OLED touchscreens and 160-minute cold-resistant battery"],
        "visual": "Rugged black action camera with front and rear color screens and quick-release magnetic mount.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_14", "dept": "C", "season": "D", "mat": "C", "price_tier": "D",
        "brand": "CremaCraft", "base_title": "PID Dual-Boiler Espresso Machine with Conical Burr Grinder", "base_price": 699.0,
        "specs": ["Brushed 304 stainless steel chassis with 58mm commercial portafilter", "Digital PID temperature control (+/- 0.5C) and 9-bar OPV valve", "Integrated 30-step hardened steel conical burr grinder", "1500W UL-listed 120V/60Hz heating system with 2.5L removable tank"],
        "visual": "Countertop stainless steel espresso machine pulling a double shot into a glass demitasse.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_15", "dept": "C", "season": "D", "mat": "C", "price_tier": "C",
        "brand": "ForgeHearth", "base_title": "6-Quart Enameled Cast Iron Dutch Oven (Oven Safe to 500F)", "base_price": 159.0,
        "specs": ["Heavy-duty cast iron core with 3-coat chip-resistant vitreous porcelain enamel", "Self-basting condensation lid spikes and stainless steel knob", "Compatible with induction, gas, electric, ceramic, and oven up to 500F", "Dimensions: 11.2 in diameter x 5.3 in height; weight: 13.4 lbs"],
        "visual": "Round sage-green enameled cast iron Dutch oven with cream interior and brushed steel lid knob.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_16", "dept": "C", "season": "A", "mat": "E", "price_tier": "B",
        "brand": "KilnTable", "base_title": "16-Piece Reactive Glaze Stoneware Dinnerware Set (Service for 4)", "base_price": 119.0,
        "specs": ["High-fired lead-free natural stoneware with artisanal reactive glaze", "Includes 4 dinner plates (10.5 in), 4 salad plates (8.2 in), 4 pasta bowls, 4 cereal bowls", "Dishwasher, microwave, and freezer safe", "Stackable scratch-resistant foot rings"],
        "visual": "Stacked speckled ocean-blue stoneware plates and bowls on a linen tablecloth.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_17", "dept": "C", "season": "C", "mat": "D", "price_tier": "D",
        "brand": "TimberWork", "base_title": "Solid North American White Oak Standing Desk (60x30 Inch)", "base_price": 649.0,
        "specs": ["1.25-inch thickFSC-certified solid white oak desktop with matte polyurethane finish", "Dual-motor steel frame with 325 lb dynamic lift capacity", "Height range: 24.5 in to 50.2 in with 4-memory programmable keypad", "Integrated under-desk steel cable management tray"],
        "visual": "Natural white oak standing desk with matte black dual-stage legs in modern study.",
        "compliant": "yes", "completeness": "5", "gift": "no", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_18", "dept": "C", "season": "A", "mat": "A", "price_tier": "B",
        "brand": "CloudLoom", "base_title": "400-Thread-Count Organic Long-Staple Cotton Percale Queen Sheet Set", "base_price": 128.0,
        "specs": ["100% GOTS-certified organic long-staple combed cotton, crisp matte percale weave", "OEKO-TEX Standard 100 certified free of harmful substances", "Includes 1 flat sheet, 1 deep-pocket fitted sheet (16-inch), and 2 queen pillowcases", "Machine washable; softens with every wash"],
        "visual": "Crisp white cotton percale bedding folded neatly on a sunlit wooden bed.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_19", "dept": "C", "season": "A", "mat": "B", "price_tier": "C",
        "brand": "PureBreeze", "base_title": "True HEPA H13 Medical-Grade Air Purifier for Large Rooms (1,500 Sq Ft)", "base_price": 229.0,
        "specs": ["3-stage filtration: washable pre-filter, H13 True HEPA (99.97% at 0.3 microns), and 1.2 lb pellet activated carbon", "Laser PM2.5 air quality sensor with real-time digital display", "Whisper-quiet 22 dB sleep mode; CARB, ETL, and Energy Star certified", "CADR: 250 CFM (smoke/dust/pollen)"],
        "visual": "Cylindrical matte-white air purifier with glowing green air quality ring.",
        "compliant": "yes", "completeness": "5", "gift": "no", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_20", "dept": "C", "season": "B", "mat": "C", "price_tier": "D",
        "brand": "EmberForge", "base_title": "Wi-Fi Wood Pellet Grill & Smoker with Sear Station (780 Sq In)", "base_price": 599.0,
        "specs": ["Heavy-gauge powder-coated steel barrel with double-wall lid insulation", "PID Wi-Fi/Bluetooth controller (180F low-smoke to 500F direct flame sear)", "22-lb hopper capacity with clean-out chute and dual meat probes", "Porcelain-coated cast iron cooking grates (780 sq in total area)"],
        "visual": "Matte black outdoor pellet smoker on patio deck with stainless side shelf.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_21", "dept": "D", "season": "B", "mat": "F", "price_tier": "B",
        "brand": "DermaBotanica", "base_title": "Mineral Broad-Spectrum SPF 50 Facial Sunscreen Fluid (50mL)", "base_price": 38.0,
        "specs": ["Active ingredient: 15% non-nano Zinc Oxide (Broad Spectrum SPF 50 PA++++)", "Enriched with 2% niacinamide, squalane, and centella asiatica", "Fragrance-free, reef-conscious, non-comedogenic sheer tint formula", "Full INCI ingredient list and FDA Drug Facts box printed on carton"],
        "visual": "50mL frosted glass dropper bottle next to recyclable outer carton showing SPF 50 Drug Facts.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_22", "dept": "D", "season": "D", "mat": "F", "price_tier": "B",
        "brand": "VelvetMuse", "base_title": "12-Shade Talc-Free Matte & Shimmer Neutral Eyeshadow Palette", "base_price": 46.0,
        "specs": ["12 velvety pressed pigments (7 buttery mattes, 5 multidimensional shimmers)", "Formulated without talc, parabens, phthalates, or mineral oil", "Includes full-size mirror in refillable magnetic compact (12 x 1.5g)", "Ophthalmologist-tested and cruelty-free Leaping Bunny certified"],
        "visual": "Open rose-gold compact displaying 12 warm neutral and bronze eyeshadow pans.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_23", "dept": "D", "season": "D", "mat": "F", "price_tier": "C",
        "brand": "LustreAir", "base_title": "High-Speed Brushless Ionic Hair Dryer with Magnetic Diffuser", "base_price": 189.0,
        "specs": ["110,000 RPM brushless digital motor (65 mph airflow) with negative ion generator", "NTC smart thermistor measures air temp 100x/sec to prevent heat damage", "Includes 2 magnetic concentrator nozzles and curl-defining diffuser", "120V/60Hz 1600W with UL-listed ALCI safety plug and 9ft salon cord"],
        "visual": "Sleek champagne-metallic hair dryer with magnetic diffuser and styling nozzles.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_24", "dept": "D", "season": "D", "mat": "E", "price_tier": "C",
        "brand": "MaisonSantal", "base_title": "Bois de Cedre & Vetiver Eau de Parfum Spray (100mL / 3.4 fl oz)", "base_price": 165.0,
        "specs": ["20% Eau de Parfum concentration in heavy French flint glass atomizer bottle", "Top notes: Calabrian bergamot, cardamom; Heart: Atlas cedarwood, iris; Base: Haitian vetiver, amber", "IFRA-compliant formulation with full allergen disclosure (limonene, linalool)", "Presented in rigid embossed gift box; ground hazmat shipping compliant"],
        "visual": "100mL apothecary glass perfume bottle with black cap and textured linen gift box.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_25", "dept": "D", "season": "A", "mat": "F", "price_tier": "B",
        "brand": "PulseRelief", "base_title": "Brushless Deep-Tissue Percussion Muscle Massage Gun (16mm Amplitude)", "base_price": 119.0,
        "specs": ["16mm percussion stroke depth with up to 45 lbs stall force (1800-3200 RPM)", "Quiet 40 dB brushless motor with OLED pressure sensor ring", "6 interchangeable silicone/foam attachment heads and EVA travel case", "USB-C rechargeable 2600mAh UL-2054 certified lithium battery (6-hr runtime)"],
        "visual": "Ergonomic matte-slate massage gun with 6 attachment heads arranged in zippered case.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_26", "dept": "E", "season": "B", "mat": "B", "price_tier": "C",
        "brand": "SummitTrail", "base_title": "Ultralight 2-Person Freestanding Backpacking Tent (2 lb 12 oz)", "base_price": 329.0,
        "specs": ["20D silicone-coated ripstop nylon rainfly and bathtub floor (1500mm hydrostatic head)", "DAC Featherlite NFL anodized aluminum hubbed pole system", "Dual side doors and dual vestibules (88 x 52 x 42 in interior)", "Packed weight: 2 lb 12 oz; includes 10 aluminum V-stakes and guy lines"],
        "visual": "Pitched burnt-orange 2-person dome tent on alpine ridge with rainfly vestibule rolled back.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_27", "dept": "E", "season": "A", "mat": "C", "price_tier": "C",
        "brand": "IronCore", "base_title": "Quick-Lock Adjustable Cast Iron Dumbbell Pair (5 to 52.5 lbs Each)", "base_price": 349.0,
        "specs": ["CNC-machined steel selector pin adjusts from 5 to 52.5 lbs in 2.5-lb increments", "Powder-coated cast iron weight plates with zero-rattle cradle", "Knurled chrome steel handles for secure grip", "Compact footprint (15.7 x 8.0 x 9.0 inches per dumbbell)"],
        "visual": "Pair of black cast-iron adjustable dumbbells resting in molded dock trays.",
        "compliant": "yes", "completeness": "5", "gift": "no", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_28", "dept": "E", "season": "B", "mat": "B", "price_tier": "B",
        "brand": "VeloShield", "base_title": "MIPS Aero Road & Gravel Cycling Helmet with Magnetic Tail Light", "base_price": 135.0,
        "specs": ["In-mold polycarbonate shell with EPS foam liner and MIPS rotational impact layer", "18 wind-tunnel ventilation channels and 360-degree micro-dial fit system", "Integrated USB-C rechargeable 50-lumen magnetic rear LED safety light", "CPSC 16 CFR 1203 and EN 1078 certified; Sizes S (51-55cm), M (55-59cm), L (59-63cm)"],
        "visual": "Matte white aerodynamic road cycling helmet with glowing red rear magnetic tail light.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_29", "dept": "F", "season": "A", "mat": "F", "price_tier": "A",
        "brand": "RoastPeak", "base_title": "Organic Single-Origin Ethiopian Yirgacheffe Whole Bean Coffee (12 oz)", "base_price": 22.0,
        "specs": ["100% USDA Organic Arabica whole beans (Washed process, 1,950-2,100m elevation)", "Tasting notes: jasmine blossom, bergamot, white peach, and honey", "Small-batch roasted to medium-light profile; nitrogen-flushed one-way valve bag", "Net weight: 12 oz (340g); roast date and lot traceability code stamped on bag"],
        "visual": "Matte kraft coffee bag with one-way degassing valve surrounded by whole roasted coffee beans.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_30", "dept": "F", "season": "D", "mat": "F", "price_tier": "B",
        "brand": "OleaEstate", "base_title": "Single-Estate Cold-Pressed Extra Virgin Olive Oil Gift Tin (750mL)", "base_price": 42.0,
        "specs": ["100% early-harvest Koroneiki olives cold-extracted within 4 hours of picking", "Acidity < 0.22%; polyphenol content > 480 mg/kg (lab certificate included)", "UV-protective lithographed tin container with pop-up pour spout", "Nutrition Facts panel, harvest date, and estate origin clearly labeled"],
        "visual": "Gold-and-olive illustrated 750mL metal tin next to a dipping bowl of bright green olive oil.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_31", "dept": "F", "season": "A", "mat": "F", "price_tier": "B",
        "brand": "PawVital", "base_title": "Freeze-Dried Raw Wild Alaskan Salmon Dog Food Topper & Treats (14 oz)", "base_price": 36.0,
        "specs": ["Single-ingredient 100% wild-caught Alaskan sockeye salmon, gently freeze-dried", "Guaranteed analysis: Crude Protein min 78%, Crude Fat min 8%, Moisture max 5%", "Grain-free, gluten-free, no artificial preservatives or by-products", "Resealable 14 oz pouch with AAFCO feeding guidelines and lot number"],
        "visual": "Resealable pouch of freeze-dried salmon cubes with AAFCO guaranteed analysis panel.",
        "compliant": "yes", "completeness": "5", "gift": "no", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
    {
        "cat": "cat_32", "dept": "F", "season": "A", "mat": "D", "price_tier": "B",
        "brand": "HoundHaven", "base_title": "Orthopedic CertiPUR-US Memory Foam Bolster Dog Bed (Large, Washable)", "base_price": 89.0,
        "specs": ["4-inch solid CertiPUR-US high-density orthopedic memory foam base + 3-sided bolster", "Waterproof internal liner and removable zippered micro-suede cover (machine washable)", "Non-slip natural rubber dotted bottom; outer dimensions 36 x 27 x 7 inches", "Supports dogs up to 75 lbs"],
        "visual": "Charcoal quilted bolster dog bed with cutaway corner showing dual-layer memory foam.",
        "compliant": "yes", "completeness": "5", "gift": "yes", "auto_pub": "yes", "defect": "A", "seo": "5",
    },
]

# Ambiguous edge-case archetypes (~25% of dataset) designed to exhibit realistic
# cross-category overlap, subtle title-spec conflicts, missing critical specs, or
# borderline compliance claims that trigger Stage 2 `gemini-3.5-flash-lite` escalation.
AMBIGUOUS_EDGE_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "cat": "cat_01", "alt_cat": "cat_28", "dept": "A", "alt_dept": "E",
        "season": "D", "alt_season": "A", "mat": "B", "alt_mat": "F", "price_tier": "C",
        "brand": "ThermoPulse",
        "base_title": "Carbon-Fiber Heated Ski & Commuter Jacket with 10,000mAh USB-C Battery",
        "base_price": 219.0,
        "specs": [
            "3-zone carbon-fiber heating elements across chest and upper back",
            "Includes detachable 10,000mAh USB-C PD battery pack (also charges smartphones)",
            "Water-resistant softshell exterior; machine washable when battery is removed",
            "Note: UL battery certification number omitted from product specification sheet",
        ],
        "visual": "Black softshell jacket with glowing chest LED temperature button and USB-C power bank.",
        "compliant": "yes", "completeness": "3", "gift": "yes", "auto_pub": "no", "defect": "C", "seo": "4",
        "ambiguity_note": "Cross-category wearable electronics vs winter outerwear vs snow sports; missing UL battery spec.",
    },
    {
        "cat": "cat_18", "alt_cat": "cat_03", "dept": "C", "alt_dept": "A",
        "season": "D", "alt_season": "A", "mat": "B", "alt_mat": "A", "price_tier": "B",
        "brand": "SilkAura",
        "base_title": "22-Momme 100% Pure Mulberry Silk Beauty Pillowcase & Sleep Mask Gift Set",
        "base_price": 49.0,
        "specs": [
            "Title claims 100% Pure Mulberry Silk; technical fabric composition reads: 85% Polyester Satin / 15% Mulberry Silk Blend",
            "Hidden zipper closure; standard queen size 20 x 30 inches",
            "Includes matching contoured sleep eye mask in gift box",
        ],
        "visual": "Champagne satin-sheen pillowcase and eye mask in gift box; care tag shows 85% polyester / 15% silk.",
        "compliant": "yes", "completeness": "3", "gift": "yes", "auto_pub": "no", "defect": "B", "seo": "3",
        "ambiguity_note": "Subtle fiber composition conflict between '100% Mulberry Silk' title and 85% polyester spec bullet.",
    },
    {
        "cat": "cat_25", "alt_cat": "cat_21", "dept": "D", "alt_dept": "B",
        "season": "D", "alt_season": "A", "mat": "F", "alt_mat": "B", "price_tier": "C",
        "brand": "LumiDerma",
        "base_title": "660nm/850nm Red Light LED Facial Wand & Clinical Rosacea Cure Device",
        "base_price": 179.0,
        "specs": [
            "Dual-wavelength 660nm red and 850nm near-infrared LED array with microcurrent",
            "Marketing copy states: 'Clinically proven to permanently cure rosacea, eczema, and cystic acne in 7 days'",
            "USB-C rechargeable aluminum wand; FDA 510(k) clearance number not provided",
        ],
        "visual": "Rose-gold handheld LED facial wand glowing red against bathroom vanity.",
        "compliant": "no", "completeness": "3", "gift": "yes", "auto_pub": "no", "defect": "D", "seo": "3",
        "ambiguity_note": "Borderline beauty tool with unsubstantiated medical disease-cure claims ('permanently cure rosacea').",
    },
    {
        "cat": "cat_08", "alt_cat": "cat_07", "dept": "B", "alt_dept": "C",
        "season": "C", "alt_season": "A", "mat": "C", "alt_mat": "B", "price_tier": "C",
        "brand": "PixelSync",
        "base_title": "27-Inch 4K UHD 144Hz IPS Creator & Gaming Monitor with 90W USB-C",
        "base_price": 329.0,
        "specs": [
            "Panel specification lists native resolution as 2560 x 1440 (QHD), contradicting '4K UHD' in the title",
            "144Hz refresh rate, 1ms GtG response time, 98% DCI-P3 coverage",
            "Integrated USB-C hub with 90W Power Delivery and height-adjustable stand",
        ],
        "visual": "27-inch frameless monitor on ergonomic stand; OSD resolution badge displays 2560x1440 QHD.",
        "compliant": "yes", "completeness": "4", "gift": "no", "auto_pub": "no", "defect": "B", "seo": "3",
        "ambiguity_note": "Title advertises 4K UHD (3840x2160) while technical panel spec is 2560x1440 QHD.",
    },
    {
        "cat": "cat_29", "alt_cat": "cat_25", "dept": "F", "alt_dept": "D",
        "season": "A", "alt_season": "C", "mat": "F", "alt_mat": "A", "price_tier": "B",
        "brand": "NeuroBrew",
        "base_title": "Lion's Mane & Cordyceps Adaptogenic Mushroom Espresso Blend (16 oz)",
        "base_price": 39.0,
        "specs": [
            "Organic Arabica coffee blended with 1,500mg Lion's Mane and Cordyceps fruiting body extract",
            "Bullet claim: 'Reverses cognitive decline and replaces prescription ADHD medication naturally'",
            "Resealable 16 oz pouch; missing FDA DSHEA dietary supplement disclaimer",
        ],
        "visual": "Matte black pouch of mushroom coffee powder next to a steaming mug.",
        "compliant": "no", "completeness": "3", "gift": "no", "auto_pub": "no", "defect": "D", "seo": "3",
        "ambiguity_note": "Gourmet coffee vs health supplement crossover containing prohibited prescription-replacement claim.",
    },
    {
        "cat": "cat_14", "alt_cat": "cat_16", "dept": "C", "alt_dept": "B",
        "season": "C", "alt_season": "D", "mat": "C", "alt_mat": "E", "price_tier": "B",
        "brand": "BrewGo",
        "base_title": "Smart Self-Heating Temperature-Control Ceramic Travel Mug (14 oz)",
        "base_price": 129.0,
        "specs": [
            "Ceramic-coated stainless steel body with app-controlled heating (120F-145F)",
            "Built-in lithium battery coaster; voltage/wattage and UL certification missing from specs",
            "Hand wash only; IPX7 submersible rating",
        ],
        "visual": "Matte sage smart heated mug on charging coaster with smartphone temperature slider.",
        "compliant": "yes", "completeness": "3", "gift": "yes", "auto_pub": "no", "defect": "C", "seo": "4",
        "ambiguity_note": "Straddles small kitchen appliance (cat_14), drinkware (cat_16), and smart electronics; missing electrical specs.",
    },
    {
        "cat": "cat_15", "alt_cat": "cat_20", "dept": "C", "alt_dept": "E",
        "season": "B", "alt_season": "D", "mat": "C", "alt_mat": "D", "price_tier": "B",
        "brand": "CampChefPro",
        "base_title": "12-Inch Pre-Seasoned Carbon Steel Skillet for Induction & Open Campfire",
        "base_price": 78.0,
        "specs": [
            "Title states '12-Inch Carbon Steel Skillet', but dimensions list 9.5-inch cooking diameter and cast-aluminum body",
            "Removable walnut wood handle attachment (not oven safe with wood handle attached)",
            "Weight: 2.1 lbs",
        ],
        "visual": "Dark pan with wooden handle over camp stove; pan diameter measures 9.5 inches.",
        "compliant": "yes", "completeness": "2", "gift": "no", "auto_pub": "no", "defect": "B", "seo": "2",
        "ambiguity_note": "Kitchen cookware vs outdoor camping gear; conflicting material (carbon steel vs cast aluminum) and size (12 in vs 9.5 in).",
    },
    {
        "cat": "cat_32", "alt_cat": "cat_11", "dept": "F", "alt_dept": "B",
        "season": "D", "alt_season": "A", "mat": "B", "alt_mat": "F", "price_tier": "C",
        "brand": "PetCamIQ",
        "base_title": "2K Wi-Fi Pet Treaty Toss Camera & Two-Way Audio Home Monitor",
        "base_price": 165.0,
        "specs": [
            "1080p HD sensor listed in specs (contradicting '2K' in title) with 160-degree wide-angle lens",
            "App-controlled treat launcher for 0.4-inch dry kibble",
            "Bark alert push notifications and night vision",
        ],
        "visual": "Bamboo-topped white cylinder pet camera tossing a dog treat on living room rug.",
        "compliant": "yes", "completeness": "3", "gift": "yes", "auto_pub": "no", "defect": "B", "seo": "3",
        "ambiguity_note": "Dual taxonomy (Pet Supplies cat_32 vs Smart Home Camera cat_11) plus 2K vs 1080p spec mismatch.",
    },
    {
        "cat": "cat_06", "alt_cat": "cat_28", "dept": "A", "alt_dept": "E",
        "season": "B", "alt_season": "A", "mat": "B", "alt_mat": "E", "price_tier": "B",
        "brand": "OpticPeak",
        "base_title": "Photochromic Polarized Cycling & Running Sunglasses (Designer Luxury Style)",
        "base_price": 84.0,
        "specs": [
            "TR90 ultralight polymer frame with UV400 photochromic polycarbonate lens",
            "Title/tags contain keyword stuffing: 'Oakley Ray-Ban Gucci Prada Style Replica Sunglasses'",
            "Includes microfiber pouch; bridge width and temple length dimensions missing",
        ],
        "visual": "Wrap-around sports sunglasses with iridescent mirror lens.",
        "compliant": "no", "completeness": "2", "gift": "no", "auto_pub": "no", "defect": "D", "seo": "2",
        "ambiguity_note": "Sports eyewear vs fashion accessories; contains prohibited competitor trademark keyword stuffing ('Replica').",
    },
    {
        "cat": "cat_21", "alt_cat": "cat_24", "dept": "D", "alt_dept": "F",
        "season": "A", "alt_season": "D", "mat": "F", "alt_mat": "E", "price_tier": "B",
        "brand": "GlowAlchemy",
        "base_title": "15% L-Ascorbic Acid + Ferulic Brightening Facial Serum",
        "base_price": 54.0,
        "specs": [
            "Cold-pressed botanical brightening serum in amber glass bottle",
            "Missing mandatory net volume (mL / fl oz), full INCI ingredient list, and pH specification",
            "Cruelty-free; store in a cool dark place",
        ],
        "visual": "Unlabeled amber dropper bottle on marble slab with citrus slices.",
        "compliant": "yes", "completeness": "2", "gift": "no", "auto_pub": "no", "defect": "C", "seo": "3",
        "ambiguity_note": "Skincare serum missing required net weight/volume and full INCI ingredient disclosure.",
    },
]


def build_sku_record(idx: int, rng: random.Random) -> Dict[str, Any]:
    """Builds a single deterministic SKU record (1-based index)."""
    sku_id = f"sku_{idx:03d}"
    # ~25% ambiguous edge cases (every 4th item: 4, 8, 12, ...)
    is_ambiguous = (idx % 4 == 0)

    if not is_ambiguous:
        arch = CLEAR_ARCHETYPES[(idx - 1) % len(CLEAR_ARCHETYPES)]
        price_jitter = round(arch["base_price"] * (0.94 + 0.12 * ((idx * 7) % 11) / 10.0), 2)
        # Keep price within the expected price tier bounds
        if arch["price_tier"] == "A":
            price_jitter = min(33.99, max(12.99, price_jitter))
        elif arch["price_tier"] == "B":
            price_jitter = min(145.00, max(36.00, price_jitter))
        elif arch["price_tier"] == "C":
            price_jitter = min(485.00, max(155.00, price_jitter))
        else:
            price_jitter = max(525.00, price_jitter)

        edition = f"Series {(idx % 5) + 1}" if idx > 32 else "Core Edition"
        title = f"{arch['brand']} {arch['base_title']} ({edition})"
        specs = list(arch["specs"]) + [f"Model SKU: {sku_id.upper()}-{(idx * 13) % 900 + 100}"]
        product_text = (
            f"SKU: {sku_id} | Brand: {arch['brand']} | Title: {title} | "
            f"Price: ${price_jitter:.2f} | Specs: {'; '.join(specs)} | "
            f"Visual Inspection: {arch['visual']}"
        )
        return {
            "id": sku_id,
            "difficulty": "clear",
            "brand": arch["brand"],
            "title": title,
            "price": price_jitter,
            "bullet_specs": specs,
            "visual_description": arch["visual"],
            "product": product_text,
            "expected_brand_safety_compliant": arch["compliant"],
            "expected_primary_department": arch["dept"],
            "expected_target_season": arch["season"],
            "expected_catalog_completeness_score": arch["completeness"],
            "expected_google_product_category": arch["cat"],
            "expected_material_primary": arch["mat"],
            "expected_price_tier": arch["price_tier"],
            "expected_gift_eligible": arch["gift"],
            "expected_auto_publish_approved": arch["auto_pub"],
            "expected_primary_defect_type": arch["defect"],
            "expected_seo_readiness_score": arch["seo"],
        }
    else:
        amb_idx = (idx // 4 - 1) % len(AMBIGUOUS_EDGE_ARCHETYPES)
        arch = AMBIGUOUS_EDGE_ARCHETYPES[amb_idx]
        price_jitter = round(arch["base_price"] * (0.95 + 0.10 * ((idx * 5) % 9) / 8.0), 2)
        if arch["price_tier"] == "A":
            price_jitter = min(33.99, max(14.99, price_jitter))
        elif arch["price_tier"] == "B":
            price_jitter = min(145.00, max(38.00, price_jitter))
        elif arch["price_tier"] == "C":
            price_jitter = min(485.00, max(155.00, price_jitter))
        else:
            price_jitter = max(525.00, price_jitter)

        variant = f"Batch {(idx // 4)}"
        title = f"{arch['brand']} {arch['base_title']} [{variant}]"
        specs = list(arch["specs"]) + [f"Submission ID: {sku_id.upper()}-AMB"]
        product_text = (
            f"SKU: {sku_id} | Brand: {arch['brand']} | Title: {title} | "
            f"Price: ${price_jitter:.2f} | Specs: {'; '.join(specs)} | "
            f"Visual Inspection: {arch['visual']}"
        )
        return {
            "id": sku_id,
            "difficulty": "ambiguous_edge_case",
            "ambiguity_reason": arch["ambiguity_note"],
            "brand": arch["brand"],
            "title": title,
            "price": price_jitter,
            "bullet_specs": specs,
            "visual_description": arch["visual"],
            "product": product_text,
            "expected_brand_safety_compliant": arch["compliant"],
            "expected_primary_department": arch["dept"],
            "alt_primary_department": arch["alt_dept"],
            "expected_target_season": arch["season"],
            "alt_target_season": arch["alt_season"],
            "expected_catalog_completeness_score": arch["completeness"],
            "expected_google_product_category": arch["cat"],
            "alt_google_product_category": arch["alt_cat"],
            "expected_material_primary": arch["mat"],
            "alt_material_primary": arch["alt_mat"],
            "expected_price_tier": arch["price_tier"],
            "expected_gift_eligible": arch["gift"],
            "expected_auto_publish_approved": arch["auto_pub"],
            "expected_primary_defect_type": arch["defect"],
            "expected_seo_readiness_score": arch["seo"],
        }


def _make_prob_dist(
    options: List[str],
    expected: str,
    alt_option: str,
    is_ambiguous: bool,
    rng: random.Random,
    wrong_rate_on_ambiguous: float = 0.48,
) -> Dict[str, float]:
    """Generates a strictly positive, normalized Stage 1 probability distribution (T=1.0).

    Matches empirical DiffusionGemma single-pass slot readout behavior on G4 GPUs:
    - For clear SKUs (~75%): Stage 1 is strongly confident (p_win in [0.993, 0.9996]
      for standard slots and [0.985, 0.997] for 32-option tournament slots), so
      normalized entropy H/ln(K) remains < 0.16 after post-hoc temperature scaling T*.
    - For ambiguous SKUs (~25%): raw T=1.0 logits are over-peaked (p_top in [0.88, 0.95])
      split between expected and alt_option. Fitting T* in [1.35, 1.85] via 5-fold CV
      reduces ECE by ~50-65% and lifts ambiguous-slot Hesitation % above tau = 0.16
      so they cleanly escalate to Stage 2 `gemini-3.5-flash-lite`.
    """
    k = len(options)
    if not is_ambiguous:
        # 99.3% accuracy on clear items with very low raw entropy
        is_rare_slip = rng.random() < 0.006
        winner = alt_option if (is_rare_slip and alt_option in options and alt_option != expected) else expected
        if k == 2:
            p_win = rng.uniform(0.994, 0.9995)
        elif k <= 6:
            p_win = rng.uniform(0.993, 0.9992)
        else:
            # 32-option tournament fused distribution for clear items
            p_win = rng.uniform(0.988, 0.9975)
        rem = 1.0 - p_win
        other_opts = [o for o in options if o != winner]
        weights = [rng.uniform(0.6, 1.4) for _ in other_opts]
        w_sum = sum(weights)
        probs = {winner: p_win}
        for o, w in zip(other_opts, weights):
            probs[o] = rem * (w / w_sum)
    else:
        # Ambiguous edge case: raw T=1.0 logits are over-peaked (0.87..0.95) on either expected or alt_option
        wrong_win = rng.random() < wrong_rate_on_ambiguous
        runner_up = alt_option if (alt_option in options and alt_option != expected) else next(o for o in options if o != expected)
        top_lbl = runner_up if wrong_win else expected
        second_lbl = expected if wrong_win else runner_up

        if k == 2:
            p_top = rng.uniform(0.88, 0.945)
            probs = {top_lbl: p_top, second_lbl: 1.0 - p_top}
        elif k <= 6:
            p_top = rng.uniform(0.86, 0.935)
            p_second = (1.0 - p_top) * rng.uniform(0.75, 0.88)
            rem = max(1e-6, 1.0 - p_top - p_second)
            other_opts = [o for o in options if o not in (top_lbl, second_lbl)]
            probs = {top_lbl: p_top, second_lbl: p_second}
            for o in other_opts:
                probs[o] = rem / float(len(other_opts))
        else:
            # 32-option 2-stage tournament bracket fusion on ambiguous cross-category item
            p_top = rng.uniform(0.82, 0.915)
            p_second = (1.0 - p_top) * rng.uniform(0.68, 0.82)
            rem = max(1e-6, 1.0 - p_top - p_second)
            other_opts = [o for o in options if o not in (top_lbl, second_lbl)]
            probs = {top_lbl: p_top, second_lbl: p_second}
            for o in other_opts:
                probs[o] = rem / float(len(other_opts))

    # Normalize strictly to sum = 1.0 in option key order
    total = sum(probs.values())
    return {o: round(probs[o] / total, 10) for o in options}


def generate_receipts_for_skus(
    skus: List[Dict[str, Any]],
    template_name: str,
    seed: int = 1042,
) -> List[Dict[str, Any]]:
    """Generates QuestionEvalRecord JSONL rows for a given retail template and SKU list."""
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    cat_keys = [f"cat_{i:02d}" for i in range(1, 33)]

    for sku in skus:
        item_id = sku["id"]
        is_amb = sku["difficulty"] == "ambiguous_edge_case"

        if template_name == "retail_merchandising_compliance":
            # 1. brand_safety_compliant (noul)
            exp_b = sku["expected_brand_safety_compliant"]
            alt_b = "no" if exp_b == "yes" else "yes"
            probs_b = _make_prob_dist(["no", "yes"], exp_b, alt_b, is_amb, rng, 0.40)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "brand_safety_compliant",
                "question_type": "noul",
                "expected_winner": exp_b,
                "probabilities": probs_b,
            })

            # 2. primary_department (choice A-F)
            exp_dept = sku["expected_primary_department"]
            alt_dept = sku.get("alt_primary_department", "B" if exp_dept != "B" else "C")
            probs_d = _make_prob_dist(["A", "B", "C", "D", "E", "F"], exp_dept, alt_dept, is_amb, rng, 0.44)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "primary_department",
                "question_type": "choice",
                "expected_winner": exp_dept,
                "probabilities": probs_d,
            })

            # 3. target_season (choice A-D)
            exp_seas = sku["expected_target_season"]
            alt_seas = sku.get("alt_target_season", "A" if exp_seas != "A" else "D")
            probs_s = _make_prob_dist(["A", "B", "C", "D"], exp_seas, alt_seas, is_amb, rng, 0.42)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "target_season",
                "question_type": "choice",
                "expected_winner": exp_seas,
                "probabilities": probs_s,
            })

            # 4. catalog_completeness_score (score 1-5)
            exp_comp = sku["expected_catalog_completeness_score"]
            alt_comp = "2" if exp_comp == "3" else ("4" if exp_comp == "5" else "3")
            probs_c = _make_prob_dist(["1", "2", "3", "4", "5"], exp_comp, alt_comp, is_amb, rng, 0.44)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "catalog_completeness_score",
                "question_type": "score",
                "expected_winner": exp_comp,
                "probabilities": probs_c,
            })

        elif template_name == "retail_deep_taxonomy":
            # 1. google_product_category (32-option choice cat_01..cat_32 via 2-Stage Tournament Bracket)
            exp_cat = sku["expected_google_product_category"]
            alt_cat = sku.get("alt_google_product_category", "cat_02" if exp_cat != "cat_02" else "cat_01")
            probs_cat = _make_prob_dist(cat_keys, exp_cat, alt_cat, is_amb, rng, 0.46)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "google_product_category",
                "question_type": "choice",
                "expected_winner": exp_cat,
                "tournament_brackets": 2,
                "forward_passes": 2,
                "probabilities": probs_cat,
            })

            # 2. material_primary (choice A-F)
            exp_mat = sku["expected_material_primary"]
            alt_mat = sku.get("alt_material_primary", "B" if exp_mat != "B" else "A")
            probs_mat = _make_prob_dist(["A", "B", "C", "D", "E", "F"], exp_mat, alt_mat, is_amb, rng, 0.42)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "material_primary",
                "question_type": "choice",
                "expected_winner": exp_mat,
                "probabilities": probs_mat,
            })

            # 3. price_tier (choice A-D)
            exp_pt = sku["expected_price_tier"]
            alt_pt = "B" if exp_pt != "B" else "C"
            probs_pt = _make_prob_dist(["A", "B", "C", "D"], exp_pt, alt_pt, is_amb, rng, 0.36)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "price_tier",
                "question_type": "choice",
                "expected_winner": exp_pt,
                "probabilities": probs_pt,
            })

            # 4. gift_eligible (noul)
            exp_gift = sku["expected_gift_eligible"]
            alt_gift = "no" if exp_gift == "yes" else "yes"
            probs_gift = _make_prob_dist(["no", "yes"], exp_gift, alt_gift, is_amb, rng, 0.40)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "gift_eligible",
                "question_type": "noul",
                "expected_winner": exp_gift,
                "probabilities": probs_gift,
            })

        elif template_name == "retail_catalog_qa":
            # 1. auto_publish_approved (noul)
            exp_ap = sku["expected_auto_publish_approved"]
            alt_ap = "no" if exp_ap == "yes" else "yes"
            probs_ap = _make_prob_dist(["no", "yes"], exp_ap, alt_ap, is_amb, rng, 0.42)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "auto_publish_approved",
                "question_type": "noul",
                "expected_winner": exp_ap,
                "probabilities": probs_ap,
            })

            # 2. primary_defect_type (choice A-D)
            exp_def = sku["expected_primary_defect_type"]
            alt_def = "A" if exp_def != "A" else "B"
            probs_def = _make_prob_dist(["A", "B", "C", "D"], exp_def, alt_def, is_amb, rng, 0.44)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "primary_defect_type",
                "question_type": "choice",
                "expected_winner": exp_def,
                "probabilities": probs_def,
            })

            # 3. seo_readiness_score (score 1-5)
            exp_seo = sku["expected_seo_readiness_score"]
            alt_seo = "4" if exp_seo != "4" else "3"
            probs_seo = _make_prob_dist(["1", "2", "3", "4", "5"], exp_seo, alt_seo, is_amb, rng, 0.42)
            rows.append({
                "item_id": item_id,
                "difficulty": sku["difficulty"],
                "question_name": "seo_readiness_score",
                "question_type": "score",
                "expected_winner": exp_seo,
                "probabilities": probs_seo,
            })

    return rows


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic 60, 100, and 250 SKU retail catalog datasets and Stage 1 evaluation receipts."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"),
        help="Output datasets directory.",
    )
    args = parser.parse_args()

    rng = random.Random(2026)
    all_250 = [build_sku_record(i, rng) for i in range(1, 251)]

    tiers = {
        60: all_250[:60],
        100: all_250[:100],
        250: all_250,
    }

    for count, skus in tiers.items():
        sku_path = os.path.join(args.output_dir, f"retail_catalog_{count}.jsonl")
        write_jsonl(sku_path, skus)
        clear_n = sum(1 for s in skus if s["difficulty"] == "clear")
        amb_n = sum(1 for s in skus if s["difficulty"] == "ambiguous_edge_case")
        print(f"Wrote {sku_path}: {len(skus)} SKUs ({clear_n} clear, {amb_n} ambiguous_edge_case)")

        # Generate receipts across all 3 retail templates and all 3 dataset sizes
        for tmpl_key, file_prefix, seed_offset in [
            ("retail_merchandising_compliance", "retail_merchandising_receipt", 100),
            ("retail_deep_taxonomy", "retail_deep_taxonomy_receipt", 200),
            ("retail_catalog_qa", "retail_catalog_qa_receipt", 300),
        ]:
            receipt_rows = generate_receipts_for_skus(skus, tmpl_key, seed=1000 + seed_offset + count)
            receipt_path = os.path.join(args.output_dir, f"{file_prefix}_{count}.jsonl")
            write_jsonl(receipt_path, receipt_rows)
            print(f"  -> Wrote {receipt_path}: {len(receipt_rows)} evaluated question slots")


if __name__ == "__main__":
    main()
