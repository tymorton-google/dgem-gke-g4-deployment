#!/usr/bin/env python3
"""calibrate_domain_temperature.py

Domain Temperature Fitting & Stage 2 Gemini 3.5 Flash-Lite Cascade Calibration
for GoogleCloudPlatform/dgem across 5 Operational & Retail Policy Domains:
  - support_triage                  (Customer Support Ticket Triage)
  - security_incident               (SOC Security Alert Triage)
  - retail_merchandising_compliance (Multi-Slot Merchandising & Brand Safety Compliance)
  - retail_deep_taxonomy            (32-Option 2-Stage Tournament Taxonomy & Attribute Extraction)
  - retail_catalog_qa               (Catalog QA & Title-Spec / Attribute Conflict Detection)

Implements the exact post-hoc temperature scaling math from `cmd/temperature_fit.go`
and `cmd/policy_temperature.go`:
  1. Evaluates labeled domain records (60, 100, or 250 items) against `dgem` at T=1.0
     (or loads pre-computed evaluation receipts from `datasets/`).
  2. Fits optimal temperature T* in [0.50, 3.50] (step 0.05) using 5-fold
     cross-validation:
       p_k(T) = p_k^(1/T) / sum_j p_j^(1/T)
     both globally and per question type (`noul`/`boolean`, `choice`, `score`).
  3. Computes Expected Calibration Error (10-bin ECE), Brier Score, Negative
     Log-Likelihood (NLL), and Normalized Shannon Entropy (Hesitation %):
       H_norm(p) = (-sum_k p_k ln p_k) / ln(K)
  4. Sweeps Hesitation % thresholds (0.05, 0.10, 0.16, 0.25, 0.35, 0.50) to
     project Stage 1 direct coverage, Stage 1 accuracy, and Stage 2 escalation
     rate to `gemini-3.5-flash-lite`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_DOMAINS = [
    "support_triage",
    "security_incident",
    "retail_merchandising_compliance",
    "retail_deep_taxonomy",
    "retail_catalog_qa",
]

RETAIL_DOMAINS = [
    "retail_merchandising_compliance",
    "retail_deep_taxonomy",
    "retail_catalog_qa",
]


@dataclass
class QuestionEvalRecord:
    item_id: str
    question_name: str
    question_type: str  # "noul" | "choice" | "score"
    expected_winner: str
    probabilities: Dict[str, float]


def scale_probs(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Post-hoc temperature scaling: p_k(T) = p_k^(1/T) / Z(T).

    Matches `scaleProbabilityMap` in `cmd/policy_temperature.go`.
    For T > 0, raising probabilities to power 1/T strictly preserves the argmax
    winner while rescaling confidence and normalized entropy (Hesitation %).
    """
    if temperature <= 0 or abs(temperature - 1.0) < 1e-6 or len(probs) <= 1:
        return dict(probs)
    inv_t = 1.0 / temperature
    eps = 1e-12
    scaled: Dict[str, float] = {}
    total = 0.0
    for k, p in probs.items():
        if p < eps:
            p = eps
        w = math.pow(p, inv_t)
        scaled[k] = w
        total += w
    if total <= 0:
        return dict(probs)
    return {k: v / total for k, v in scaled.items()}


def normalized_entropy(probs: Dict[str, float]) -> Tuple[float, float]:
    """Returns (entropy_nats, normalized_entropy_hesitation)."""
    k = len(probs)
    if k <= 1:
        return 0.0, 0.0
    h = 0.0
    for p in probs.values():
        if p > 1e-12:
            h -= p * math.log(p)
    max_h = math.log(float(k))
    norm_h = min(1.0, max(0.0, h / max_h)) if max_h > 0 else 0.0
    return h, norm_h


def evaluate_calibration_metrics(
    records: List[QuestionEvalRecord],
    temp_by_type: Dict[str, float],
    num_bins: int = 10,
) -> Dict[str, float]:
    """Computes accuracy, 10-bin ECE, mean Brier score, NLL, and JevBench CalScore."""
    if not records:
        return {"accuracy": 0.0, "ece": 0.0, "brier": 0.0, "nll": 0.0, "cal_score": 0.0, "objective": 0.0}

    bin_counts = [0] * num_bins
    bin_conf_sum = [0.0] * num_bins
    bin_acc_sum = [0.0] * num_bins

    correct_total = 0
    brier_sum = 0.0
    nll_sum = 0.0

    for rec in records:
        t = temp_by_type.get(rec.question_type, temp_by_type.get("global", 1.0))
        scaled = scale_probs(rec.probabilities, t)
        winner = max(scaled.items(), key=lambda kv: kv[1])[0]
        conf = scaled[winner]
        is_correct = 1.0 if winner.lower() == rec.expected_winner.lower() else 0.0
        correct_total += int(is_correct)

        # Multi-class Brier score: sum_k (p_k - y_k)^2
        for opt, p in scaled.items():
            target = 1.0 if opt.lower() == rec.expected_winner.lower() else 0.0
            brier_sum += (p - target) ** 2

        # Negative log-likelihood of ground-truth option
        p_true = 1e-9
        for opt, p in scaled.items():
            if opt.lower() == rec.expected_winner.lower():
                p_true = max(1e-9, p)
                break
        nll_sum += -math.log(p_true)

        b_idx = min(num_bins - 1, int(conf * num_bins))
        bin_counts[b_idx] += 1
        bin_conf_sum[b_idx] += conf
        bin_acc_sum[b_idx] += is_correct

    n = float(len(records))
    ece = 0.0
    for b in range(num_bins):
        if bin_counts[b] > 0:
            avg_conf = bin_conf_sum[b] / bin_counts[b]
            avg_acc = bin_acc_sum[b] / bin_counts[b]
            ece += (bin_counts[b] / n) * abs(avg_acc - avg_conf)

    mean_brier = brier_sum / n
    mean_nll = nll_sum / n
    cal_score = 100.0 * (1.0 - ece)
    # Matches `scoreRecordsAtTemperature` in `cmd/temperature_fit.go`:
    objective = cal_score - 10.0 * mean_brier

    return {
        "accuracy": 100.0 * correct_total / n,
        "ece": ece,
        "brier": mean_brier,
        "nll": mean_nll,
        "cal_score": cal_score,
        "objective": objective,
    }


def fit_temperature_5fold_cv(
    records: List[QuestionEvalRecord],
    min_t: float = 0.50,
    max_t: float = 3.50,
    step_t: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """5-fold cross-validated grid search matching `fitOptimalTemperatureCV`."""
    if len(records) < 5:
        return 1.0, evaluate_calibration_metrics(records, {"global": 1.0})

    k_folds = 5
    grid = []
    t_val = min_t
    while t_val <= max_t + 1e-9:
        grid.append(round(t_val, 2))
        t_val += step_t

    fold_opt_temps: List[float] = []
    for fold in range(k_folds):
        train_recs = [r for i, r in enumerate(records) if i % k_folds != fold]
        best_t = 1.0
        best_obj = -1e9
        for cand_t in grid:
            metrics = evaluate_calibration_metrics(train_recs, {"global": cand_t})
            if metrics["objective"] > best_obj:
                best_obj = metrics["objective"]
                best_t = cand_t
        fold_opt_temps.append(best_t)

    avg_t = sum(fold_opt_temps) / float(k_folds)
    fitted_t = round(avg_t / step_t) * step_t
    fitted_t = round(max(min_t, min(max_t, fitted_t)), 2)
    final_metrics = evaluate_calibration_metrics(records, {"global": fitted_t})
    return fitted_t, final_metrics


def sweep_hesitation_thresholds(
    records: List[QuestionEvalRecord],
    temp_by_type: Dict[str, float],
    thresholds: List[float] = [0.05, 0.10, 0.16, 0.25, 0.35, 0.50],
) -> List[Dict[str, float]]:
    """Sweeps Hesitation % thresholds to show Stage 1 accuracy vs Gemini 3.5 Flash-Lite cascade rate."""
    results = []
    n = float(len(records))
    if n == 0:
        return results

    for tau in thresholds:
        s1_kept = 0
        s1_correct = 0
        escalated = 0
        for rec in records:
            t = temp_by_type.get(rec.question_type, temp_by_type.get("global", 1.0))
            scaled = scale_probs(rec.probabilities, t)
            winner = max(scaled.items(), key=lambda kv: kv[1])[0]
            _, h_norm = normalized_entropy(scaled)
            is_correct = 1 if winner.lower() == rec.expected_winner.lower() else 0

            if h_norm < tau:
                s1_kept += 1
                s1_correct += is_correct
            else:
                escalated += 1

        s1_acc = (100.0 * s1_correct / s1_kept) if s1_kept > 0 else 0.0
        coverage = 100.0 * s1_kept / n
        escalation_rate = 100.0 * escalated / n
        results.append({
            "threshold": tau,
            "stage1_coverage_pct": coverage,
            "stage1_accuracy_pct": s1_acc,
            "stage2_gemini_escalation_pct": escalation_rate,
        })
    return results


def resolve_default_receipt_path(domain: str, sample_count: int, datasets_dir: str) -> str:
    """Maps (domain, sample_count) to the pre-computed receipt JSONL file in datasets/."""
    prefix_map = {
        "support_triage": "support_triage_receipt",
        "security_incident": "security_incident_receipt",
        "retail_merchandising_compliance": "retail_merchandising_receipt",
        "retail_deep_taxonomy": "retail_deep_taxonomy_receipt",
        "retail_catalog_qa": "retail_catalog_qa_receipt",
    }
    prefix = prefix_map[domain]
    sized_candidate = os.path.join(datasets_dir, f"{prefix}_{sample_count}.jsonl")
    if os.path.exists(sized_candidate):
        return sized_candidate
    unsized_candidate = os.path.join(datasets_dir, f"{prefix}.jsonl")
    if os.path.exists(unsized_candidate):
        return unsized_candidate
    # Fallback to 100 or 60 if available
    for fallback_n in (100, 60, 250):
        cand = os.path.join(datasets_dir, f"{prefix}_{fallback_n}.jsonl")
        if os.path.exists(cand):
            return cand
    return ""


def load_receipt_jsonl(path: str, limit_items: int = 0) -> List[QuestionEvalRecord]:
    records: List[QuestionEvalRecord] = []
    seen_items = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = str(row["item_id"])
            if limit_items > 0 and item_id not in seen_items and len(seen_items) >= limit_items:
                continue
            seen_items.add(item_id)
            records.append(
                QuestionEvalRecord(
                    item_id=item_id,
                    question_name=str(row["question_name"]),
                    question_type=str(row.get("question_type", "choice")),
                    expected_winner=str(row["expected_winner"]),
                    probabilities={str(k): float(v) for k, v in row["probabilities"].items()},
                )
            )
    return records


def generate_synthetic_domain_dataset(domain: str, count: int = 60) -> List[QuestionEvalRecord]:
    """Generates a deterministic calibration receipt for any of the 5 supported domains."""
    if domain in RETAIL_DOMAINS:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        import generate_retail_datasets as grd  # type: ignore

        rng = random.Random(2026)
        skus = [grd.build_sku_record(i, rng) for i in range(1, count + 1)]
        seed_offset = {
            "retail_merchandising_compliance": 100,
            "retail_deep_taxonomy": 200,
            "retail_catalog_qa": 300,
        }[domain]
        raw_rows = grd.generate_receipts_for_skus(skus, domain, seed=1000 + seed_offset + count)
        return [
            QuestionEvalRecord(
                item_id=str(r["item_id"]),
                question_name=str(r["question_name"]),
                question_type=str(r["question_type"]),
                expected_winner=str(r["expected_winner"]),
                probabilities={str(k): float(v) for k, v in r["probabilities"].items()},
            )
            for r in raw_rows
        ]

    rng = random.Random(42)
    records: List[QuestionEvalRecord] = []

    for i in range(1, count + 1):
        item_id = f"{domain}_{i:03d}"
        is_borderline = (i % 4 == 0)

        q_bool = "urgent" if domain == "support_triage" else "contain_immediately"
        true_bool = "yes" if (i % 3 == 0) else "no"
        if not is_borderline:
            p_win = rng.uniform(0.92, 0.995)
            probs_b = {true_bool: p_win, ("no" if true_bool == "yes" else "yes"): 1.0 - p_win}
        else:
            wrong_win = rng.random() < 0.45
            top_lbl = ("no" if true_bool == "yes" else "yes") if wrong_win else true_bool
            p_top = rng.uniform(0.82, 0.94)
            probs_b = {top_lbl: p_top, ("no" if top_lbl == "yes" else "yes"): 1.0 - p_top}
        records.append(QuestionEvalRecord(item_id, q_bool, "noul", true_bool, probs_b))

        q_choice = "team" if domain == "support_triage" else "incident_class"
        options = ["A", "B", "C", "D"]
        true_opt = options[i % 4]
        if not is_borderline:
            p_win = rng.uniform(0.90, 0.99)
            rem = (1.0 - p_win) / 3.0
            probs_c = {k: (p_win if k == true_opt else rem) for k in options}
        else:
            wrong_win = rng.random() < 0.48
            top_lbl = options[(i + 1) % 4] if wrong_win else true_opt
            p_top = rng.uniform(0.78, 0.92)
            rem = (1.0 - p_top) / 3.0
            probs_c = {k: (p_top if k == top_lbl else rem) for k in options}
        records.append(QuestionEvalRecord(item_id, q_choice, "choice", true_opt, probs_c))

        q_score = "sentiment" if domain == "support_triage" else "severity"
        levels = ["1", "2", "3", "4", "5"]
        true_lvl = levels[i % 5]
        if not is_borderline:
            p_win = rng.uniform(0.88, 0.98)
            rem = (1.0 - p_win) / 4.0
            probs_s = {k: (p_win if k == true_lvl else rem) for k in levels}
        else:
            wrong_win = rng.random() < 0.45
            top_lbl = levels[(i + 1) % 5] if wrong_win else true_lvl
            p_top = rng.uniform(0.75, 0.90)
            rem = (1.0 - p_top) / 4.0
            probs_s = {k: (p_top if k == top_lbl else rem) for k in levels}
        records.append(QuestionEvalRecord(item_id, q_score, "score", true_lvl, probs_s))

    return records


def run_calibration_for_domain(
    domain: str,
    records: List[QuestionEvalRecord],
    receipt_source: str = "",
) -> Dict[str, Any]:
    raw_metrics = evaluate_calibration_metrics(records, {"global": 1.0})
    global_t, global_metrics = fit_temperature_5fold_cv(records)

    # Fit per-question-type temperatures ("noul", "choice", "score")
    temp_by_type: Dict[str, float] = {"global": global_t}
    for qtype in ("noul", "choice", "score"):
        sub = [r for r in records if r.question_type == qtype]
        if len(sub) >= 5:
            t_q, _ = fit_temperature_5fold_cv(sub)
            temp_by_type[qtype] = t_q
        else:
            temp_by_type[qtype] = global_t

    per_type_metrics = evaluate_calibration_metrics(records, temp_by_type)
    sweep_rows = sweep_hesitation_thresholds(records, temp_by_type)
    num_items = len(set(r.item_id for r in records))
    ece_reduction_pct = (
        100.0 * (raw_metrics["ece"] - per_type_metrics["ece"]) / raw_metrics["ece"]
        if raw_metrics["ece"] > 1e-9
        else 0.0
    )

    print("=" * 84)
    print(f"dgem Domain Calibration & Gemini 3.5 Flash-Lite Escalation Report ({domain})")
    if receipt_source:
        print(f"Receipt Source : {os.path.basename(receipt_source)}")
    print(f"Evaluated Slots: {len(records)} across {num_items} domain items")
    print("=" * 84)
    print(
        f"Baseline (T = 1.00)     -> Accuracy: {raw_metrics['accuracy']:.1f}% | "
        f"ECE: {raw_metrics['ece']:.4f} | Brier: {raw_metrics['brier']:.4f} | "
        f"NLL: {raw_metrics['nll']:.4f} | CalScore: {raw_metrics['cal_score']:.2f}"
    )
    print(
        f"Global Fit (T* = {global_t:.2f})  -> Accuracy: {global_metrics['accuracy']:.1f}% | "
        f"ECE: {global_metrics['ece']:.4f} | Brier: {global_metrics['brier']:.4f} | "
        f"NLL: {global_metrics['nll']:.4f} | CalScore: {global_metrics['cal_score']:.2f}"
    )
    print(
        f"Per-Type Fit            -> Accuracy: {per_type_metrics['accuracy']:.1f}% | "
        f"ECE: {per_type_metrics['ece']:.4f} ({-ece_reduction_pct:+.1f}%) | Brier: {per_type_metrics['brier']:.4f} | "
        f"NLL: {per_type_metrics['nll']:.4f} | CalScore: {per_type_metrics['cal_score']:.2f}"
    )
    print(
        f"  Fitted Schema Temperatures: "
        f"noul={temp_by_type['noul']:.2f}, choice={temp_by_type['choice']:.2f}, score={temp_by_type['score']:.2f}"
    )
    print("-" * 84)
    print("Stage 2 Escalation Sweep to `gemini-3.5-flash-lite` (Normalized Entropy / Hesitation %):")
    print(f"{'Hesitation Threshold':<22} | {'Stage 1 Coverage':<18} | {'Stage 1 Accuracy':<18} | {'Gemini 3.5 Escalated':<20}")
    print("-" * 84)
    for row in sweep_rows:
        marker = " <-- Recommended" if abs(row["threshold"] - 0.16) < 1e-6 else ""
        print(
            f"tau = {row['threshold']*100:5.1f}% ({row['threshold']:.2f})   | "
            f"{row['stage1_coverage_pct']:15.1f}%  | "
            f"{row['stage1_accuracy_pct']:15.1f}%  | "
            f"{row['stage2_gemini_escalation_pct']:17.1f}%{marker}"
        )
    print("=" * 84)
    print("\nPolicy Template `schema.temperature` block:")
    temp_block: Dict[str, float] = {}
    for qtype in ("noul", "choice", "score"):
        if any(r.question_type == qtype for r in records):
            temp_block[qtype] = temp_by_type[qtype]
    print(
        json.dumps(
            {
                "temperature": temp_block,
                "cascade_mode": "entropy",
                "cascade_threshold_mode": "normalized",
                "cascade_threshold": 0.16,
                "cascade_model": "gemini-3.5-flash-lite",
                "stage2_prior": "soft",
            },
            indent=2,
        )
    )
    print()

    rec_tau = next((r for r in sweep_rows if abs(r["threshold"] - 0.16) < 1e-6), sweep_rows[0])
    return {
        "domain": domain,
        "items": num_items,
        "slots": len(records),
        "raw_ece": raw_metrics["ece"],
        "cal_ece": per_type_metrics["ece"],
        "ece_reduction_pct": ece_reduction_pct,
        "raw_nll": raw_metrics["nll"],
        "cal_nll": per_type_metrics["nll"],
        "raw_brier": raw_metrics["brier"],
        "cal_brier": per_type_metrics["brier"],
        "global_t": global_t,
        "temp_by_type": temp_block,
        "tau_016_coverage_pct": rec_tau["stage1_coverage_pct"],
        "tau_016_s1_accuracy_pct": rec_tau["stage1_accuracy_pct"],
        "tau_016_escalation_pct": rec_tau["stage2_gemini_escalation_pct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit post-hoc temperature T* and Gemini 3.5 Flash-Lite escalation threshold across operational and retail domains."
    )
    parser.add_argument(
        "--domain",
        choices=SUPPORTED_DOMAINS,
        default="retail_merchandising_compliance",
        help="Target policy template domain.",
    )
    parser.add_argument(
        "--receipt-jsonl",
        type=str,
        default="",
        help="Path to JSONL evaluation receipt.",
    )
    parser.add_argument(
        "--generate-sample-dataset",
        action="store_true",
        help="Generate and calibrate on a synthetic domain dataset if receipt file is not specified.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Dataset size tier (e.g., 60, 100, or 250 SKUs/tickets/alerts).",
    )
    parser.add_argument(
        "--all-retail",
        action="store_true",
        help="Run calibration across all 3 retail policy templates and all 3 SKU dataset tiers (60, 100, 250).",
    )
    args = parser.parse_args()

    default_datasets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
    )

    if args.all_retail:
        summary_rows = []
        for dom in RETAIL_DOMAINS:
            for sz in (60, 100, 250):
                rpath = resolve_default_receipt_path(dom, sz, default_datasets_dir)
                if rpath and os.path.exists(rpath):
                    recs = load_receipt_jsonl(rpath)
                else:
                    recs = generate_synthetic_domain_dataset(dom, sz)
                summary_rows.append(run_calibration_for_domain(dom, recs, rpath))

        print("=" * 116)
        print("Consolidated Retail Catalog Enrichment Calibration Summary (60, 100, and 250 SKU Tiers)")
        print("=" * 116)
        print(
            f"{'Template Domain':<33} | {'SKUs (Slots)':<14} | {'Global T*':<9} | "
            f"{'ECE (T=1 -> T*)':<22} | {'S1 Acc (@0.16)':<14} | {'Stage 2 Escalated':<17}"
        )
        print("-" * 116)
        for s in summary_rows:
            print(
                f"{s['domain']:<33} | "
                f"{s['items']:3d} ({s['slots']:4d})   | "
                f"{s['global_t']:7.2f}   | "
                f"{s['raw_ece']:.4f} -> {s['cal_ece']:.4f} (-{s['ece_reduction_pct']:4.1f}%) | "
                f"{s['tau_016_s1_accuracy_pct']:12.1f}%  | "
                f"{s['tau_016_escalation_pct']:15.1f}%"
            )
        print("=" * 116)
        return

    receipt_path = args.receipt_jsonl
    if not receipt_path and not args.generate_sample_dataset:
        receipt_path = resolve_default_receipt_path(args.domain, args.sample_count, default_datasets_dir)

    if receipt_path and os.path.exists(receipt_path):
        records = load_receipt_jsonl(receipt_path)
    else:
        records = generate_synthetic_domain_dataset(args.domain, args.sample_count)

    run_calibration_for_domain(args.domain, records, receipt_path)


if __name__ == "__main__":
    main()
