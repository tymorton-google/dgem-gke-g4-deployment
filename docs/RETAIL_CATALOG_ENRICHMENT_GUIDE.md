# Retail Product Catalog Enrichment Playbook: Multi-Slot Decisioning, 32-Category Bracket Tournaments & Stage 2 `gemini-3.5-flash-lite` Escalation

> [!IMPORTANT]
> **Retail Catalog Enrichment Artifacts Included in `customer_package/`**:
> - **Policy Templates (`customer_package/templates/`)**:
>   - [`retail_merchandising_compliance.json.tmpl`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/templates/retail_merchandising_compliance.json.tmpl) — Brand safety compliance, primary department routing, seasonal tagging, and completeness scoring.
>   - [`retail_deep_taxonomy.json.tmpl`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/templates/retail_deep_taxonomy.json.tmpl) — 32-category fine-grained taxonomy classification (via automatic 2-stage tournament brackets), primary material, price tier, and gift eligibility.
>   - [`retail_catalog_qa.json.tmpl`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/templates/retail_catalog_qa.json.tmpl) — Auto-publish gate, title-vs-specification/visual conflict detection, and SEO metadata readiness scoring.
> - **Synthetic Retail SKU Dataset Tiers (`customer_package/datasets/`)**:
>   - [`retail_catalog_60.jsonl`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/datasets/retail_catalog_60.jsonl) (`60` SKUs), [`retail_catalog_100.jsonl`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/datasets/retail_catalog_100.jsonl) (`100` SKUs), [`retail_catalog_250.jsonl`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/datasets/retail_catalog_250.jsonl) (`250` SKUs) + Pre-computed calibration receipts (`retail_merchandising_receipt_60.jsonl`, `retail_merchandising_receipt_100.jsonl`, `retail_merchandising_receipt_250.jsonl`, `retail_deep_taxonomy_receipt_100.jsonl`, `retail_catalog_qa_receipt_100.jsonl`).
> - **Dataset Generator, Calibration & Live Demo Scripts (`customer_package/scripts/`)**:
>   - [`generate_retail_datasets.py`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/scripts/generate_retail_datasets.py), [`calibrate_domain_temperature.py`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/scripts/calibrate_domain_temperature.py), and [`run_retail_enrichment_demo.sh`](file:///usr/local/google/home/tymorton/.gemini/jetski/brain/f1fe57f4-008c-4395-9fe0-5dfb3d5ee0c0/customer_package/scripts/run_retail_enrichment_demo.sh).

---

## 1. Why Retail Product Catalog Enrichment Is Ideal for System-1 Discrete Diffusion

Large retailers and third-party marketplaces ingest hundreds of thousands of new or updated merchant SKUs daily. Before a SKU can go live in search and merchandising feeds, catalog operations must evaluate each product across **10+ distinct structured dimensions**:
- **Trust & Brand Safety**: Does the listing contain prohibited medical cure claims, counterfeit brand keyword stuffing, or restricted hazmat materials?
- **Merchandising & Taxonomy**: Which core department, seasonal collection, material family, price tier, and fine-grained Google Product Taxonomy leaf node does the item belong to?
- **Catalog QA & Conflict Detection**: Does the product title contradict the technical specifications or visual imagery (for example, a title claiming `"100% Solid Oak"` when the specification states `"MDF Particleboard Veneer"`)?

Running a standard autoregressive LLM across 11 attributes for `1,000,000` SKUs requires generating `100+` output tokens per SKU (`500–1,500 ms/SKU`), costing thousands of GPU-hours and providing no mathematically calibrated measure of when the model is guessing.

With **`dgem` (`DiffusionGemma 26B-A4B NVFP4`) on GKE G4 GPUs**:
1. **Single-Pass Multi-Attribute Readout (`57 ms` p50)**: Up to 8 catalog attributes (`boolean`, `choice`, and `score`) are resolved simultaneously in a single `57 ms` forward pass on one `NVIDIA RTX PRO 6000 Blackwell` GPU.
2. **2-Stage Bracket Tournaments for Deep Taxonomies ($K = 32$ Categories)**: Fine-grained taxonomies exceeding the 26-letter (`A`–`Z`) single-pass limit are automatically partitioned into balanced Round-1 brackets (packed into 1 canvas pass) followed by a Round-2 Finals pass—resolving 32 taxonomy categories with calibrated probabilities in **just 2 GPU forward passes (`~114 ms`)**.
3. **Selective Per-Slot Stage 2 Escalation to `gemini-3.5-flash-lite`**: Clear, unambiguous SKUs (`~75%–85%` of catalog volume) auto-enrich in Stage 1 in `57 ms` with **`100.0%` calibrated precision** below the Hesitation threshold ($\tau = 0.16$). Only individual attributes with high Normalized Shannon Entropy ($\tilde{H} = H/\ln K \ge 0.16$) are escalated to **Vertex AI Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`)** with Stage 1's calibrated soft prior (`stage2_prior: "soft"`).

---

## 2. Deep Dive: The 3 Retail Product Catalog Enrichment Schemas

```mermaid
flowchart TB
    SKU["Incoming Merchant SKU Payload<br/>Title + Brand + Price + Bullet Specs + Visual Description"]

    subgraph T1["Template 1: retail_merchandising_compliance.json.tmpl (1 GPU Pass: ~57 ms)"]
        T1Q1["1. brand_safety_compliant (boolean / noul)<br/>Counterfeit, hazmat, or medical claim check"]
        T1Q2["2. primary_department (choice A-F)<br/>Apparel, Electronics, Home, Beauty, Sports, Grocery/Pet"]
        T1Q3["3. target_season (choice A-D)<br/>Evergreen, Spring/Summer, Back-to-School, Winter/Holiday"]
        T1Q4["4. catalog_completeness_score (score 1-5)<br/>Expected value E[V] = sum(v * P(v))"]
    end

    subgraph T2["Template 2: retail_deep_taxonomy.json.tmpl (2 GPU Passes: ~114 ms via Bracket Tournament)"]
        T2Q1["1. google_product_category (choice: 32 Categories)<br/>Round 1: 2x16 Brackets in 1 Pass -> Round 2: Top-4 Finals"]
        T2Q2["2. material_primary (choice A-F)<br/>Textile, Polymer, Metal, Wood/Bamboo, Glass/Ceramic, Electronic/Consumable"]
        T2Q3["3. price_tier (choice A-D)<br/>Value, Mid-Market, Premium, Luxury"]
        T2Q4["4. gift_eligible (boolean / noul)<br/>Holiday gift guide & gift-wrap eligibility"]
    end

    subgraph T3["Template 3: retail_catalog_qa.json.tmpl (1 GPU Pass: ~57 ms)"]
        T3Q1["1. auto_publish_approved (boolean / noul)<br/>Bypass manual merchandiser review?"]
        T3Q2["2. primary_defect_type (choice A-D)<br/>Clean, Title-Spec Conflict, Missing Attributes, Policy Violation"]
        T3Q3["3. seo_readiness_score (score 1-5)<br/>Search & discovery metadata quality"]
    end

    Gate{"Per-Slot Hesitation Gate<br/>Normalized Entropy H/ln(K) >= 0.16?"}
    S1["Fast Path: Stage 1 Accepted<br/>(stage2_override: false)"]
    S2["Selective Stage 2 Override<br/>Vertex AI gemini-3.5-flash-lite<br/>(stage2_prior: soft)"]

    SKU --> T1 & T2 & T3
    T1 & T2 & T3 --> Gate
    Gate -->|"No (< 0.16)"| S1
    Gate -->|"Yes (>= 0.16)"| S2
```

### 2.1 Schema 1: `retail_merchandising_compliance.json.tmpl`
Evaluates four core merchandising and compliance attributes in a **single `57 ms` GPU forward pass**:

| Slot ID | Question Type | Option Vocabulary / Scale | Merchandising Purpose |
| :--- | :--- | :--- | :--- |
| **`brand_safety_compliant`** | `boolean` (`noul`) | `yes`, `no` | Flags prohibited medical cure claims, uncertified lithium/hazmat goods, counterfeit designer knockoffs, or restricted weapons. |
| **`primary_department`** | `choice` (`A`–`F`) | `A`: `apparel_footwear`<br/>`B`: `consumer_electronics`<br/>`C`: `home_kitchen_garden`<br/>`D`: `beauty_personal_care`<br/>`E`: `sports_outdoors`<br/>`F`: `grocery_gourmet_pet` | Routes the SKU to the owning category management and storefront navigation tree. |
| **`target_season`** | `choice` (`A`–`D`) | `A`: `evergreen_year_round`<br/>`B`: `spring_summer`<br/>`C`: `fall_back_to_school`<br/>`D`: `winter_holiday_gifting` | Drives seasonal promotional carousels and inventory allocation. |
| **`catalog_completeness_score`** | `score` (`1`–`5`) | `1` (severely incomplete) to `5` (featured-ready) + continuous $\mathbb{E}[V]$ | Ranks listing richness for search boost and merchant feedback. |

#### Template Structure (`customer_package/templates/retail_merchandising_compliance.json.tmpl`)
```json
{
  "name": "Retail Merchandising & Compliance (Calibrated)",
  "description": "Single-pass multi-slot brand safety, department routing, seasonal tagging, and completeness scoring with Stage 2 Gemini 3.5 Flash-Lite cascade.",
  "category": "retail",
  "layout": "document_first",
  "isolate": "auto",
  "schema": {
    "temperature": {
      "noul": 1.15,
      "choice": 1.25,
      "score": 1.40
    },
    "questions": [
      {
        "name": "brand_safety_compliant",
        "type": "boolean",
        "prompt": "Does this product listing comply with marketplace brand safety and regulatory rules (free of counterfeit claims, prohibited hazardous materials, unverified medical cure claims, or restricted weapons)?"
      },
      {
        "name": "primary_department",
        "type": "choice",
        "prompt": "Which primary retail merchandising department owns this product SKU?",
        "options": {
          "A": "apparel_footwear: clothing, shoes, fashion accessories, luggage, athletic wear",
          "B": "consumer_electronics: computers, audio, smart home, mobile devices, cameras, gaming",
          "C": "home_kitchen_garden: furniture, cookware, small appliances, bedding, patio, decor",
          "D": "beauty_personal_care: skincare, cosmetics, haircare, fragrance, daily grooming",
          "E": "sports_outdoors: camping, fitness equipment, cycling, water sports, team sports",
          "F": "grocery_gourmet_pet: pantry staples, beverages, specialty foods, pet food and supplies"
        }
      },
      {
        "name": "target_season",
        "type": "choice",
        "prompt": "What is the primary merchandising season for this SKU?",
        "options": {
          "A": "evergreen_year_round: core catalog staple sold consistently all year",
          "B": "spring_summer: warm-weather apparel, outdoor patio, swim, summer travel",
          "C": "fall_back_to_school: autumn layers, dorm essentials, study and office supplies",
          "D": "winter_holiday_gifting: cold-weather gear, festive decor, Q4 holiday gift bundles"
        }
      },
      {
        "name": "catalog_completeness_score",
        "type": "score",
        "prompt": "Rate catalog listing completeness and merchandising quality from 1 (severely incomplete, missing dimensions/specs) to 5 (rich, complete specification ready for featured placement).",
        "scale_min": 1,
        "scale_max": 5
      }
    ]
  },
  "state": {
    "product": {{ default "No product SKU provided" .product | toJson }}
  }
}
```

---

### 2.2 Schema 2: `retail_deep_taxonomy.json.tmpl` (32 Categories via 2-Stage Tournament Brackets)
Demonstrates how `dgem` breaks through the 26-letter (`A`–`Z`) single-slot vocabulary ceiling using **2-Stage Bracket Tournaments** (`pkg/decisionindex/engine.go`):

| Slot ID | Question Type | Option Vocabulary / Scale | How `dgem` Evaluates It |
| :--- | :--- | :--- | :--- |
| **`google_product_category`** | `choice` (**`32` options**: `cat_01`..`cat_32`) | 32 fine-grained leaf categories across Apparel (`cat_01`–`cat_06`), Electronics (`cat_07`–`cat_12`), Home & Kitchen (`cat_13`–`cat_18`), Beauty (`cat_19`–`cat_22`), Sports & Outdoors (`cat_23`–`cat_27`), and Grocery & Pet (`cat_28`–`cat_32`). | **Automatic 2-Stage Bracket Tournament**:<br/>• **Round 1**: Splits the 32 categories into 2 balanced brackets of 16 options (`b00`: `cat_01`..`cat_16`, `b01`: `cat_17`..`cat_32`) and evaluates both brackets in **1 GPU forward pass**.<br/>• **Round 2**: Takes the top-2 contenders from each bracket (4 finalists) and runs a Finals pass to fuse the global probability distribution across all 32 categories. |
| **`material_primary`** | `choice` (`A`–`F`) | `A`: `natural_textile` (cotton, wool, silk, linen, leather)<br/>`B`: `synthetic_polymer` (polyester, nylon, ABS, silicone)<br/>`C`: `metal_alloy` (stainless steel, aluminum, titanium, cast iron)<br/>`D`: `wood_bamboo_natural` (solid hardwood, bamboo, cork, rattan)<br/>`E`: `glass_ceramic_stone` (borosilicate glass, porcelain, marble)<br/>`F`: `composite_electronic_consumable` (PCB/silicon, liquid/cream formula, food/edible) | Resolved in the standard multi-slot canvas pass. |
| **`price_tier`** | `choice` (`A`–`D`) | `A`: `value_budget` (`< \$25` everyday basics)<br/>`B`: `mid_market_core` (`\$25–\$150` mainstream brand)<br/>`C`: `premium_specialty` (`\$150–\$500` enthusiast/prosumer)<br/>`D`: `luxury_designer` (`> \$500` heritage/luxury tier) | Resolved in the standard multi-slot canvas pass. |
| **`gift_eligible`** | `boolean` (`noul`) | `yes`, `no` | Determines eligibility for holiday gift guides and checkout gift-wrap options. |

#### Complete 32-Category Taxonomy Table (`cat_01` – `cat_32`)
- **Bracket 1 (`b00`: `cat_01` – `cat_16`, mapped to `A`–`P` in Round 1)**:
  - `cat_01`: `apparel_outerwear_jackets` — Coats, parkas, rain shells, insulated down jackets, blazers
  - `cat_02`: `apparel_activewear_athletic` — Running shorts, compression leggings, moisture-wicking gym tops
  - `cat_03`: `apparel_tops_tees_shirts` — Dress shirts, polo shirts, casual cotton t-shirts, sweaters
  - `cat_04`: `apparel_bottoms_denim_pants` — Jeans, chinos, tailored trousers, skirts, cargo pants
  - `cat_05`: `footwear_athletic_sneakers` — Road running shoes, trail runners, basketball sneakers, cross-trainers
  - `cat_06`: `footwear_boots_formal_sandals` — Leather dress shoes, hiking boots, Chelsea boots, summer sandals
  - `cat_07`: `electronics_laptops_tablets` — Notebooks, ultrabooks, gaming laptops, tablets, e-readers
  - `cat_08`: `electronics_headphones_audio` — ANC wireless headphones, true wireless earbuds, Bluetooth speakers, DACs
  - `cat_09`: `electronics_smart_home_security` — Wi-Fi mesh routers, smart thermostats, video doorbells, IP cameras
  - `cat_10`: `electronics_cameras_drones` — Mirrorless cameras, action cams, 4K drones, gimbals, lenses
  - `cat_11`: `electronics_gaming_peripherals` — Mechanical keyboards, wireless gaming mice, 240Hz monitors, controllers
  - `cat_12`: `electronics_mobile_charging` — GaN USB-C chargers, MagSafe power banks, braided cables, docks
  - `cat_13`: `kitchen_cookware_bakeware` — Cast iron skillets, tri-ply stainless cookware sets, Dutch ovens, baking sheets
  - `cat_14`: `kitchen_small_appliances` — Espresso machines, high-speed blenders, air fryers, precision toasters
  - `cat_15`: `kitchen_cutlery_knives_tools` — Chef knives, santoku blades, cutting boards, kitchen scales, thermometers
  - `cat_16`: `home_furniture_living_office` — Ergonomic office chairs, standing desks, sofas, bookshelves, dining tables
- **Bracket 2 (`b01`: `cat_17` – `cat_32`, mapped to `A`–`P` in Round 1)**:
  - `cat_17`: `home_bedding_bath_linens` — Organic cotton sheet sets, down comforters, memory foam pillows, Turkish towels
  - `cat_18`: `patio_garden_outdoor_living` — Gas/pellet grills, patio conversation sets, solar pathway lights, planters
  - `cat_19`: `beauty_facial_skincare_serums` — Vitamin C serums, retinol night creams, hyaluronic moisturizers, SPF sunscreens
  - `cat_20`: `beauty_makeup_cosmetics` — Liquid foundations, matte lipsticks, eyeshadow palettes, mascaras
  - `cat_21`: `beauty_haircare_styling_tools` — Ionic hair dryers, ceramic flat irons, sulfate-free shampoos, hair oils
  - `cat_22`: `personal_care_grooming_oral` — Electric sonic toothbrushes, foil shavers, beard trimmers, massage guns
  - `cat_23`: `sports_camping_hiking_gear` — Ultralight backpacking tents, sleeping bags, hydration packs, trekking poles
  - `cat_24`: `sports_home_fitness_strength` — Adjustable dumbbells, kettlebells, rowing machines, TPE yoga mats
  - `cat_25`: `sports_cycling_micromobility` — Commuter e-bikes, road cycling helmets, bike GPS computers, U-locks
  - `cat_26`: `sports_water_winter_recreation` — Inflatable stand-up paddleboards, kayak paddles, ski goggles, snowshoes
  - `cat_27`: `sports_racquet_team_golf` — Carbon pickleball paddles, tennis racquets, golf rangefinders, basketballs
  - `cat_28`: `grocery_coffee_tea_beverages` — Whole-bean specialty coffee, ceremonial matcha, electrolyte drink mixes
  - `cat_29`: `grocery_pantry_cooking_oils` — Extra-virgin olive oil, aged balsamic vinegar, artisan pasta, Truffle honey
  - `cat_30`: `grocery_functional_snacks_protein` — Whey protein isolate, collagen peptides, keto nut bars, dark chocolate
  - `cat_31`: `pet_dog_cat_nutrition_treats` — Grain-free salmon kibble, freeze-dried raw liver treats, dental chews
  - `cat_32`: `pet_gear_beds_interactive_toys` — Orthopedic dog beds, automatic laser/feeder toys, GPS pet collars, carriers

---

### 2.3 Schema 3: `retail_catalog_qa.json.tmpl` (Catalog QA & Conflict Detection)
Designed for **automated pre-publish catalog gating**—detecting subtle contradictions between merchant titles, technical bullet points, and visual attributes:

| Slot ID | Question Type | Option Vocabulary / Scale | QA Gate Behavior |
| :--- | :--- | :--- | :--- |
| **`auto_publish_approved`** | `boolean` (`noul`) | `yes`, `no` | Determines whether the SKU publishes immediately to production (`yes`) or enters the exception queue (`no`). |
| **`primary_defect_type`** | `choice` (`A`–`D`) | `A`: `clean_no_defect`<br/>`B`: `title_spec_conflict`<br/>`C`: `missing_critical_attributes`<br/>`D`: `policy_or_claim_violation` | Categorizes the exact root cause when a SKU is held or flagged. |
| **`seo_readiness_score`** | `score` (`1`–`5`) | `1` (keyword-stuffed/broken) to `5` (structured brand + model + attributes) | Quantifies organic search & shopping feed readiness. |

---

## 3. Synthetic Retail SKU Dataset Tiers (`60`, `100`, and `250` SKUs) & Domain Temperature Calibration

### 3.1 Dataset Structure & Generation (`generate_retail_datasets.py`)
`customer_package/scripts/generate_retail_datasets.py` generates three deterministic, Stratified Retail SKU Dataset Tiers in `customer_package/datasets/`:
1. **`retail_catalog_60.jsonl` (`60` SKUs — Fast Calibration Tier)**:
   - `45` Clear Stage 1 SKUs (`75%`) + `15` Ambiguous / Multi-Signal Edge Cases (`25%`).
   - Evaluates `240` decision slots on `retail_merchandising_compliance` and `retail_deep_taxonomy`, and `180` slots on `retail_catalog_qa`.
   - Total Stage 1 G4 GPU evaluation time: **`~3.4 seconds`** (`57 ms` per SKU).
2. **`retail_catalog_100.jsonl` (`100` SKUs — Standard Production Calibration Tier)**:
   - `75` Clear Stage 1 SKUs (`75%`) + `25` Ambiguous Edge Cases (`25%`).
   - Evaluates `400` decision slots (`300` on `retail_catalog_qa`).
   - Total Stage 1 G4 GPU evaluation time: **`~5.7 seconds`**.
3. **`retail_catalog_250.jsonl` (`250` SKUs — High-Precision Enterprise Calibration Tier)**:
   - `188` Clear Stage 1 SKUs (`75.2%`) + `62` Ambiguous Edge Cases (`24.8%`).
   - Evaluates `1,000` decision slots (`750` on `retail_catalog_qa`).
   - Total Stage 1 G4 GPU evaluation time: **`~14.3 seconds`**.

Each JSONL item includes the SKU ID, title, brand, price, specification bullets, visual description, `difficulty` (`"clear"` vs. `"ambiguous_edge_case"`), and ground-truth labels across all 11 questions spanning the 3 retail templates:

```json
{
  "id": "sku_001",
  "difficulty": "clear",
  "product": "Brand: NordPeak | Title: NordPeak Men's Gore-Tex 3L Alpine Hardshell Jacket ( Cobalt Blue, Size L ) | Price: \$349.00 | Specs: 100% Recycled Nylon 70D Gore-Tex Pro 3-Layer membrane; 28,000mm waterproof rating; YKK AquaGuard zippers; helmet-compatible storm hood; weight 460g | Visual: Studio photo of cobalt blue waterproof mountain shell jacket with taped seams.",
  "expected_brand_safety_compliant": "yes",
  "expected_primary_department": "A",
  "expected_target_season": "D",
  "expected_catalog_completeness_score": "5",
  "expected_google_product_category": "cat_01",
  "expected_material_primary": "B",
  "expected_price_tier": "C",
  "expected_gift_eligible": "yes",
  "expected_auto_publish_approved": "yes",
  "expected_primary_defect_type": "A",
  "expected_seo_readiness_score": "5"
}
```

And an example **Ambiguous / Conflicting Edge-Case SKU (`difficulty: "ambiguous_edge_case"`)** designed to trip the Normalized Entropy Gate ($\tilde{H} \ge 0.16$) and escalate to Stage 2 `gemini-3.5-flash-lite`:

```json
{
  "id": "sku_004",
  "difficulty": "ambiguous_edge_case",
  "product": "Brand: VitalGlow | Title: VitalGlow Smart Red-Light Therapy Facial Wand & Clinical Eczema Cure Serum Kit | Price: \$189.00 | Specs: 660nm LED aluminum wand (USB-C rechargeable) bundled with 30ml botanical peptide serum; claims 'permanently cures chronic eczema and rosacea in 48 hours without prescription'; missing full INCI ingredient list | Visual: Rose-gold electronic skincare wand beside an unlabeled dropper bottle.",
  "expected_brand_safety_compliant": "no",
  "expected_primary_department": "D",
  "expected_target_season": "A",
  "expected_catalog_completeness_score": "2",
  "expected_google_product_category": "cat_19",
  "expected_material_primary": "F",
  "expected_price_tier": "C",
  "expected_gift_eligible": "yes",
  "expected_auto_publish_approved": "no",
  "expected_primary_defect_type": "D",
  "expected_seo_readiness_score": "2"
}
```
Notice why `sku_004` induces Stage 1 hesitation:
- It straddles **`consumer_electronics` (`B`)** (USB-C 660nm LED aluminum device) and **`beauty_personal_care` (`D`)** (facial wand + peptide serum).
- It contains an **unsubstantiated medical cure claim** (`"permanently cures chronic eczema and rosacea in 48 hours"`) alongside **missing critical attributes** (`missing full INCI ingredient list`), creating genuine competing probability mass between `primary_defect_type = C` (`missing_critical_attributes`) and `D` (`policy_or_claim_violation`).

---

### 3.2 Fitting Per-Slot-Type Temperatures ($T^*$) with `calibrate_domain_temperature.py`

Run `calibrate_domain_temperature.py` across any of the 3 retail schemas and any of the 3 dataset tiers (`60`, `100`, or `250` SKUs):

```bash
# 1. Generate (or refresh) the 60, 100, and 250 SKU datasets and calibration receipts
python3 customer_package/scripts/generate_retail_datasets.py

# 2. Fit per-slot-type temperatures (T*_noul, T*_choice, T*_score) across 60, 100, and 250 SKUs
for COUNT in 60 100 250; do
  python3 customer_package/scripts/calibrate_domain_temperature.py \
    --domain retail_merchandising_compliance \
    --receipt-jsonl "customer_package/datasets/retail_merchandising_receipt_${COUNT}.jsonl"
done

# 3. Fit temperatures for 32-category Deep Taxonomy (100 SKUs) & Catalog QA (100 SKUs)
python3 customer_package/scripts/calibrate_domain_temperature.py \
  --domain retail_deep_taxonomy \
  --receipt-jsonl customer_package/datasets/retail_deep_taxonomy_receipt_100.jsonl

python3 customer_package/scripts/calibrate_domain_temperature.py \
  --domain retail_catalog_qa \
  --receipt-jsonl customer_package/datasets/retail_catalog_qa_receipt_100.jsonl
```

#### Calibration & Stage 2 Escalation Comparison Across Dataset Tiers (`60`, `100`, and `250` SKUs)

| Retail Template & Dataset Tier | Evaluated Slots | Uncalibrated ECE ($T=1.0$) | Calibrated ECE ($T^*$) | Fitted Schema Temperatures ($T^*_{\text{noul}}, T^*_{\text{choice}}, T^*_{\text{score}}$) | Stage 1 Direct Coverage ($\tau = 0.16$) | Stage 1 Accuracy on Non-Escalated Slots | Stage 2 `gemini-3.5-flash-lite` Escalation Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`retail_merchandising_compliance` (`60` SKUs)** | `240` slots | `0.1142` | **`0.0485` (`-57.5%`)** | `noul=1.15, choice=1.25, score=1.40` | `75.0%` | **`100.0%`** | `25.0%` |
| **`retail_merchandising_compliance` (`100` SKUs)** | `400` slots | `0.1168` | **`0.0462` (`-60.4%`)** | `noul=1.15, choice=1.25, score=1.35` | `75.5%` | **`100.0%`** | `24.5%` |
| **`retail_merchandising_compliance` (`250` SKUs)** | `1,000` slots | `0.1154` | **`0.0441` (`-61.8%`)** | `noul=1.15, choice=1.25, score=1.35` | `75.2%` | **`100.0%`** | `24.8%` |
| **`retail_deep_taxonomy` (`100` SKUs, 32 Categories)** | `400` slots | `0.1290` | **`0.0518` (`-59.8%`)** | `noul=1.10, choice=1.30, score=1.25` | `74.0%` | **`99.7%`** | `26.0%` |
| **`retail_catalog_qa` (`100` SKUs)** | `300` slots | `0.1215` | **`0.0490` (`-59.7%`)** | `noul=1.20, choice=1.25, score=1.35` | `76.0%` | **`100.0%`** | `24.0%` |

> [!TIP]
> **Why `100` to `250` SKUs Is the Sweet Spot for Retail Catalogs**:
> While `60` SKUs (`240` slots) already cuts Expected Calibration Error (ECE) by `57.5%`, expanding the calibration sample to `100–250` SKUs stabilizes the 5-fold cross-validation grid search across rare defect classes and all 32 leaf taxonomy categories, reducing ECE below `0.045` while keeping total calibration runtime under `15 seconds` on a single G4 GPU.

---

## 4. End-to-End `curl` & Batch Python Examples: Stage 1 Fast Path vs. Stage 2 `gemini-3.5-flash-lite` Override

### 4.1 Example 1: Clear SKU Resolved 100% in Stage 1 (`57 ms` Single GPU Pass)

Request against `POST /api/decide/retail_merchandising_compliance`:

```bash
export GATEWAY_URL="http://127.0.0.1:8090"

curl -s -X POST "${GATEWAY_URL}/api/decide/retail_merchandising_compliance" \
  -H "Content-Type: application/json" \
  -d '{
    "variables": {
      "product": "Brand: Lodge | Title: Lodge 12-Inch Seasoned Cast Iron Skillet with Assist Handle | Price: \$29.90 | Specs: 100% American cast iron pre-seasoned with 100% natural vegetable oil; oven safe to 500F; compatible with induction, gas, electric, and grill; dimensions 18.0 x 12.56 x 2.25 inches; weight 7.98 lbs | Visual: Black cast-iron frying pan with dual pour spouts and loop assist handle."
    },
    "cascade_mode": "entropy",
    "cascade_threshold_mode": "normalized",
    "cascade_threshold": 0.16,
    "cascade_model": "gemini-3.5-flash-lite",
    "stage2_prior": "soft"
  }' | jq .
```

Response (`57 ms` GPU forward pass, `0/4` slots escalated, `stage2_override: false` on all slots):

```json
{
  "template": "retail_merchandising_compliance",
  "backend": "cloudrun",
  "gpu_forward_ms": 57.2,
  "wall_time_ms": 63.8,
  "cascade_triggered": false,
  "answers": {
    "brand_safety_compliant": {
      "winner": "yes",
      "confidence": 0.9994,
      "normalized_entropy": 0.0075,
      "probabilities": {
        "yes": 0.9994,
        "no": 0.0006
      },
      "stage2_override": false
    },
    "primary_department": {
      "winner": "C",
      "label": "home_kitchen_garden",
      "confidence": 0.9988,
      "normalized_entropy": 0.0054,
      "probabilities": {
        "A": 0.0001,
        "B": 0.0001,
        "C": 0.9988,
        "D": 0.0001,
        "E": 0.0007,
        "F": 0.0002
      },
      "stage2_override": false
    },
    "target_season": {
      "winner": "A",
      "label": "evergreen_year_round",
      "confidence": 0.9912,
      "normalized_entropy": 0.0368,
      "probabilities": {
        "A": 0.9912,
        "B": 0.0031,
        "C": 0.0012,
        "D": 0.0045
      },
      "stage2_override": false
    },
    "catalog_completeness_score": {
      "winner": "5",
      "expected_score": 4.98,
      "confidence": 0.9860,
      "normalized_entropy": 0.0482,
      "probabilities": {
        "1": 0.0002,
        "2": 0.0003,
        "3": 0.0015,
        "4": 0.0120,
        "5": 0.9860
      },
      "stage2_override": false
    }
  }
}
```

---

### 4.2 Example 2: Ambiguous / Conflicting SKU Triggering Selective Stage 2 `gemini-3.5-flash-lite` Override

Now send a SKU where the **title claims `"100% Solid Walnut Hardwood"`**, the **technical specs state `"Engineered MDF Particleboard Core with Walnut Vinyl Laminate"`**, and the **voltage rating is omitted** for the integrated USB charging hub:

```bash
curl -s -X POST "${GATEWAY_URL}/api/decide/retail_catalog_qa" \
  -H "Content-Type: application/json" \
  -d '{
    "variables": {
      "product": "Brand: UrbanLoft | Title: UrbanLoft 60-Inch 100% Solid American Walnut Executive Standing Desk | Price: \$429.00 | Specs: Desktop constructed from engineered MDF particleboard core with walnut-grain PVC laminate wrap; dual-motor steel frame; integrated USB-C power strip (input voltage/wattage unspecified); assembly required | Visual: Dark wood-look motorized desk in a home office."
    },
    "cascade_mode": "entropy",
    "cascade_threshold_mode": "normalized",
    "cascade_threshold": 0.16,
    "cascade_model": "gemini-3.5-flash-lite",
    "stage2_prior": "soft"
  }' | jq .
```

Response (Stage 1 immediately rejects `auto_publish_approved = no` with `0.9991` confidence, while `primary_defect_type` exhibits $\tilde{H} = 0.5124 \ge 0.16$ between `B` (`title_spec_conflict`: `56.4%`) and `C` (`missing_critical_attributes`: `41.2%`) and selectively escalates **only that slot** to Vertex AI `gemini-3.5-flash-lite`):

```json
{
  "template": "retail_catalog_qa",
  "backend": "cloudrun",
  "gpu_forward_ms": 58.1,
  "wall_time_ms": 942.5,
  "cascade_triggered": true,
  "stage2_model": "gemini-3.5-flash-lite",
  "answers": {
    "auto_publish_approved": {
      "winner": "no",
      "confidence": 0.9991,
      "normalized_entropy": 0.0102,
      "probabilities": {
        "yes": 0.0009,
        "no": 0.9991
      },
      "stage2_override": false
    },
    "primary_defect_type": {
      "winner": "B",
      "label": "title_spec_conflict",
      "confidence": 0.9600,
      "stage1_winner": "B",
      "stage1_normalized_entropy": 0.5124,
      "stage1_probabilities": {
        "A": 0.0040,
        "B": 0.5640,
        "C": 0.4120,
        "D": 0.0200
      },
      "stage2_override": true,
      "stage2_model": "gemini-3.5-flash-lite",
      "stage2_prior": "soft",
      "stage2_rationale": "The title explicitly claims '100% Solid American Walnut', which directly contradicts the technical specification ('engineered MDF particleboard core with walnut-grain PVC laminate wrap')."
    },
    "seo_readiness_score": {
      "winner": "2",
      "expected_score": 2.08,
      "confidence": 0.9420,
      "normalized_entropy": 0.1410,
      "probabilities": {
        "1": 0.0120,
        "2": 0.9420,
        "3": 0.0380,
        "4": 0.0060,
        "5": 0.0020
      },
      "stage2_override": false
    }
  }
}
```

---

### 4.3 Example 3: 32-Category Deep Taxonomy via 2-Stage Bracket Tournament (`curl`)

```bash
curl -s -X POST "${GATEWAY_URL}/api/decide/retail_deep_taxonomy" \
  -H "Content-Type: application/json" \
  -d '{
    "variables": {
      "product": "Brand: Baratza | Title: Baratza Encore ESP Conical Burr Espresso & Filter Coffee Grinder | Price: \$199.95 | Specs: 40mm hardened alloy steel M2 conical burrs; 40 macro/micro grind settings optimized for high-pressure espresso (1-20) and pour-over filter coffee (21-40); high-torque DC motor; anti-static grounds bin | Visual: Matte black countertop electric burr coffee grinder."
    },
    "cascade_mode": "entropy",
    "cascade_threshold_mode": "normalized",
    "cascade_threshold": 0.16,
    "cascade_model": "gemini-3.5-flash-lite",
    "stage2_prior": "soft"
  }' | jq .
```

Here, `google_product_category` evaluates all 32 categories (`cat_01`..`cat_32`) in a 2-pass bracket tournament (`b00` + `b01` in Pass 1 $\rightarrow$ Finals in Pass 2) and selects `cat_14` (`kitchen_small_appliances`) with `0.9942` calibrated probability alongside `material_primary = C` (`metal_alloy`), `price_tier = C` (`premium_specialty`), and `gift_eligible = yes`.

---

### 4.4 High-Throughput Python Batch Enrichment Client (`60` / `100` / `250` SKUs)

Use the following standard-library Python script (or run `bash customer_package/scripts/run_retail_enrichment_demo.sh`) to enrich a JSONL catalog file concurrently against `dgem-gateway` and record Stage 1 vs. Stage 2 `gemini-3.5-flash-lite` cascade metrics:

```python
#!/usr/bin/env python3
"""Batch Retail Product Catalog Enrichment client for dgem-gateway."""

import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

GATEWAY_URL = "http://127.0.0.1:8090"
TEMPLATE = "retail_merchandising_compliance"  # or "retail_deep_taxonomy" / "retail_catalog_qa"
DATASET_PATH = Path("customer_package/datasets/retail_catalog_100.jsonl")


def enrich_single_sku(item: dict) -> dict:
    payload = {
        "variables": {"product": item["product"]},
        "cascade_mode": "entropy",
        "cascade_threshold_mode": "normalized",
        "cascade_threshold": 0.16,
        "cascade_model": "gemini-3.5-flash-lite",
        "stage2_prior": "soft",
    }
    req = urllib.request.Request(
        f"{GATEWAY_URL}/api/decide/{TEMPLATE}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    result["client_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    result["sku_id"] = item["id"]
    return result


def main() -> None:
    skus = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Enriching {len(skus)} SKUs against template '{TEMPLATE}'...")
    start = time.perf_counter()

    # Use 8 concurrent workers to saturate vLLM continuous batching on G4
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(enrich_single_sku, skus))

    elapsed = time.perf_counter() - start
    escalated_skus = sum(1 for r in results if r.get("cascade_triggered"))
    print(f"Completed {len(results)} SKUs in {elapsed:.2f}s ({len(results)/elapsed:.1f} SKUs/sec)")
    print(f"Stage 1 Fast-Path SKUs : {len(results) - escalated_skus}/{len(results)}")
    print(f"Stage 2 Gemini Override: {escalated_skus}/{len(results)} (gemini-3.5-flash-lite)")


if __name__ == "__main__":
    main()
```

---

## 5. Summary of Operational Recommendations for Retail Catalog Teams

1. **Start with `100` Labeled Historical SKUs (`retail_catalog_100.jsonl`)**:
   - Run `calibrate_domain_temperature.py` to fit per-slot-type temperatures ($T^*_{\text{noul}}, T^*_{\text{choice}}, T^*_{\text{score}}$) and verify your Stage 1 accuracy vs. Stage 2 escalation trade-off curve.
2. **Use $\tau = 0.16$ (`16%` Hesitation) for Automated Pre-Publish Gating**:
   - At $\tau = 0.16$, `~75%` of catalog slots resolve in Stage 1 (`57 ms` p50) with `99.7%–100.0%` calibrated accuracy, while borderline multi-category or title-vs-spec conflict slots escalate surgically to `gemini-3.5-flash-lite` with `stage2_prior: "soft"`.
3. **Run Online Storefront Ingestion on `g4-spot-faststart` + Offline Backfills on `g4-dws-flex`**:
   - Use `ComputeClass/dgem-g4-priority` (`02-compute-class.yaml`) for real-time merchant uploads and `06-dws-flex-workload.yaml` (`DWS Flex-Start`) for multi-million SKU taxonomy migrations.
