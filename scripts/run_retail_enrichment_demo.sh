#!/usr/bin/env bash
# ==============================================================================
# run_retail_enrichment_demo.sh
#
# Live interactive & batch benchmark script for the 3 Retail Product Catalog
# Enrichment policy templates on GKE G4 GPUs + Vertex AI Gemini 3.5 Flash-Lite:
#   1. POST /api/decide/retail_merchandising_compliance
#   2. POST /api/decide/retail_deep_taxonomy (32 categories via 2-stage brackets)
#   3. POST /api/decide/retail_catalog_qa
#
# Usage:
#   # Start port-forward to dgem-gateway (if not already exposed via LoadBalancer):
#   kubectl port-forward svc/dgem-gateway -n dgem 8090:80 &
#
#   # Run live single-SKU showcase (Clear Stage 1 SKU vs Ambiguous Stage 2 SKU)
#   # plus a batch benchmark across the synthetic retail catalog:
#   ./run_retail_enrichment_demo.sh \
#     --gateway-url http://127.0.0.1:8090 \
#     --dataset-tier 100 \
#     --batch-limit 10 \
#     --cascade-threshold 0.16 \
#     --cascade-model gemini-3.5-flash-lite
# ==============================================================================
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8090}"
DATASET_TIER="${DATASET_TIER:-100}" # 60 | 100 | 250
BATCH_LIMIT="${BATCH_LIMIT:-10}"    # Set to 0 to run the entire dataset tier
CASCADE_THRESHOLD="${CASCADE_THRESHOLD:-0.16}"
CASCADE_MODEL="${CASCADE_MODEL:-gemini-3.5-flash-lite}"
STAGE2_PRIOR="${STAGE2_PRIOR:-soft}"
OFFLINE_ONLY="${OFFLINE_ONLY:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -u, --gateway-url <URL>        dgem-gateway HTTP URL (default: http://127.0.0.1:8090)
  -d, --dataset-tier <60|100|250> Synthetic retail SKU dataset tier (default: 100)
  -n, --batch-limit <N>          Number of SKUs to evaluate in live batch benchmark (0 = all, default: 10)
      --cascade-threshold <TAU>  Normalized Shannon entropy (Hesitation %) threshold in [0,1] (default: 0.16)
      --cascade-model <MODEL>    Stage 2 Vertex AI Gemini model (default: gemini-3.5-flash-lite)
      --stage2-prior <MODE>      Stage 1 prior passed to Stage 2: soft | full | none (default: soft)
      --offline-calibration      Run offline temperature fitting across 60, 100, and 250 SKU receipts only
  -h, --help                     Show this help message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--gateway-url)        GATEWAY_URL="${2%/}"; shift 2 ;;
    -d|--dataset-tier)       DATASET_TIER="$2"; shift 2 ;;
    -n|--batch-limit)        BATCH_LIMIT="$2"; shift 2 ;;
    --cascade-threshold)     CASCADE_THRESHOLD="$2"; shift 2 ;;
    --cascade-model)         CASCADE_MODEL="$2"; shift 2 ;;
    --stage2-prior)          STAGE2_PRIOR="$2"; shift 2 ;;
    --offline-calibration)   OFFLINE_ONLY="1"; shift ;;
    -h|--help)               usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_FILE="${PACKAGE_DIR}/datasets/retail_catalog_${DATASET_TIER}.jsonl"

echo "=========================================================================================="
echo " dgem Retail Product Catalog Enrichment — System-1 Diffusion + Stage 2 Gemini Cascade"
echo "   Gateway URL       : ${GATEWAY_URL}"
echo "   SKU Dataset Tier  : retail_catalog_${DATASET_TIER}.jsonl (${DATASET_TIER} SKUs)"
echo "   Stage 2 Cascade   : ${CASCADE_MODEL} (Hesitation threshold tau = ${CASCADE_THRESHOLD}, prior = ${STAGE2_PRIOR})"
echo "=========================================================================================="

echo ""
echo "==> [Step 1/3] Running Multi-Tier Domain Temperature Calibration (60, 100, and 250 SKUs)..."
python3 "${SCRIPT_DIR}/calibrate_domain_temperature.py" --all-retail

if [[ "${OFFLINE_ONLY}" == "1" ]]; then
  echo "Offline calibration complete."
  exit 0
fi

echo ""
echo "==> [Step 2/3] Checking Live dgem-gateway Readiness at ${GATEWAY_URL}/api/templates..."
if ! curl -sf --max-time 5 "${GATEWAY_URL}/api/templates" >/dev/null 2>&1; then
  echo "WARNING: Could not reach ${GATEWAY_URL}/api/templates."
  echo "To run live GKE G4 GPU queries, ensure dgem-engine is scaled to 1 and port-forward dgem-gateway:"
  echo "  kubectl scale deployment dgem-engine -n dgem --replicas=1"
  echo "  kubectl port-forward svc/dgem-gateway -n dgem 8090:80 &"
  exit 0
fi

echo ""
echo "==> [Step 3/3] Executing Live Queries & Batch Benchmark Across All 3 Retail Templates..."
export GATEWAY_URL DATASET_FILE BATCH_LIMIT CASCADE_THRESHOLD CASCADE_MODEL STAGE2_PRIOR

python3 - <<'EOF'
import json
import math
import os
import statistics
import time
import urllib.request
import urllib.error

gateway_url = os.environ["GATEWAY_URL"]
dataset_file = os.environ["DATASET_FILE"]
batch_limit = int(os.environ["BATCH_LIMIT"])
cascade_threshold = float(os.environ["CASCADE_THRESHOLD"])
cascade_model = os.environ["CASCADE_MODEL"]
stage2_prior = os.environ["STAGE2_PRIOR"]

with open(dataset_file, "r", encoding="utf-8") as f:
    skus = [json.loads(line) for line in f if line.strip()]

clear_sku = next(s for s in skus if s["difficulty"] == "clear")
ambig_sku = next(s for s in skus if s["difficulty"] == "ambiguous_edge_case")

templates = [
    "retail_merchandising_compliance",
    "retail_deep_taxonomy",
    "retail_catalog_qa",
]

def call_decide(template_name: str, product_text: str) -> dict:
    url = f"{gateway_url}/api/decide/{template_name}"
    payload = {
        "variables": {"product": product_text},
        "cascade_mode": "entropy",
        "cascade_threshold_mode": "normalized",
        "cascade_threshold": cascade_threshold,
        "cascade_model": cascade_model,
        "stage2_prior": stage2_prior,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("\n------------------------------------------------------------------------------------------")
print(" Live Showcase A: Clear Stage 1 SKU vs Ambiguous Edge-Case SKU (Stage 2 Escalation)")
print("------------------------------------------------------------------------------------------")
for label, sample_sku in [("CLEAR STAGE 1 SKU", clear_sku), ("AMBIGUOUS EDGE-CASE SKU", ambig_sku)]:
    print(f"\n[{label}] {sample_sku['id']} — {sample_sku['title']}")
    for tmpl in templates:
        t0 = time.time()
        res = call_decide(tmpl, sample_sku["product"])
        wall_ms = int((time.time() - t0) * 1000)
        gpu_ms = res.get("gpu_forward_ms", wall_ms)
        cascade = res.get("cascade") or {}
        trig = cascade.get("triggered", False)
        esc_cnt = cascade.get("escalated_count", 0)
        ans_summary = []
        for qid, qans in (res.get("answers") or {}).items():
            choice = qans.get("choice") or qans.get("value") or qans.get("noul") or qans.get("score")
            conf = qans.get("confidence", 0.0)
            s2 = " [Stage2 Override]" if qans.get("stage2_override") else ""
            ans_summary.append(f"{qid}={choice} ({conf*100:.1f}%{s2})")
        print(
            f"  • {tmpl:<32} | GPU: {gpu_ms:4d} ms | Wall: {wall_ms:4d} ms | "
            f"Stage 2 Triggered: {str(trig):<5} ({esc_cnt} slots) | {', '.join(ans_summary)}"
        )

eval_skus = skus if batch_limit <= 0 else skus[:batch_limit]
print("\n------------------------------------------------------------------------------------------")
print(f" Live Batch Benchmark Summary ({len(eval_skus)} SKUs across all 3 Retail Policy Templates)")
print("------------------------------------------------------------------------------------------")
print(f"{'Retail Policy Template':<34} | {'SKUs':<5} | {'GPU p50 (ms)':<12} | {'GPU Mean (ms)':<13} | {'Stage 2 Escalated SKUs':<22}")
print("-" * 96)

for tmpl in templates:
    gpu_times = []
    escalated_skus = 0
    for s in eval_skus:
        res = call_decide(tmpl, s["product"])
        gpu_times.append(int(res.get("gpu_forward_ms", 0)))
        if (res.get("cascade") or {}).get("triggered"):
            escalated_skus += 1
    p50_ms = int(statistics.median(gpu_times)) if gpu_times else 0
    mean_ms = statistics.mean(gpu_times) if gpu_times else 0.0
    print(
        f"{tmpl:<34} | {len(eval_skus):<5d} | {p50_ms:10d} ms | {mean_ms:10.1f} ms | "
        f"{escalated_skus:4d} / {len(eval_skus):<4d} ({100.0*escalated_skus/max(1,len(eval_skus)):5.1f}%)"
    )
print("------------------------------------------------------------------------------------------")
EOF
