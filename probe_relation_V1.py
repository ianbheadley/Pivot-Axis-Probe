"""
probe_relation.py — General Relational Structure Probe
=======================================================
Reads ALL domain definitions from an external JSON file.
No domain logic lives in this script.

Three validation gates per domain:
  Gate A: Spearman rho significant (p < 0.05)
  Gate B: Delta-rho vs null axis > 0.30
  Gate C: Intruder variance ratio > 3x AND no intruder within 20% of poles

Built on vb_core.py — requires all three vb_core gates to pass first.

Usage:
    python probe_relation.py                          # run all domains
    python probe_relation.py --category physical      # filter by category
    python probe_relation.py --keys biological_scale  # specific domain(s)
    python probe_relation.py --resume                 # skip completed
    python probe_relation.py --list                   # list domains and exit
    python probe_relation.py --layer 20               # different layer

Output:
    probe_results.json   -- full results, written after every domain
    probe_summary.txt    -- human-readable ranked summary
"""

import json
import argparse
import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from mlx_lm import load

# IMPORT THE FOUNDATION GATES
from vb_core import (
    get_concept_vector, 
    get_axis_vector,
    gate_1_extractor_sanity,
    gate_2_geographic_ordering,
    gate_3_null_model
)

# -----------------------------------------------------------------------------
# GATE C: INTRUDER VALIDATION
# -----------------------------------------------------------------------------

def run_intruder_check(intruders, axis, model, tokenizer, layer,
                       in_scores, cat_min, cat_max):
    if not intruders:
        return None

    pole_range   = cat_max - cat_min
    lo_edge      = cat_min + 0.20 * pole_range
    hi_edge      = cat_max - 0.20 * pole_range

    out_scores   = {}
    pole_biased  = []

    print(f"\n  Gate C: Intruder Check ({len(intruders)} out-of-domain concepts)")
    print(f"  Category range: [{cat_min:+.4f}, {cat_max:+.4f}]")
    print(f"  Alarm zone:  score < {lo_edge:+.4f}  or  score > {hi_edge:+.4f}")
    print(f"  {'Concept':<24} {'Score':>8}  Status")
    print(f"  {'─'*46}")

    for c in intruders:
        vec   = get_concept_vector(c, model, tokenizer, layer)
        score = float(vec @ axis)
        out_scores[c] = score

        if score < lo_edge:
            status = "ALARM: low pole"
            pole_biased.append({"concept": c, "score": score, "pole": "low"})
        elif score > hi_edge:
            status = "ALARM: high pole"
            pole_biased.append({"concept": c, "score": score, "pole": "high"})
        else:
            status = "null zone ok"

        print(f"  {c:<24} {score:>+8.4f}  {status}")

    in_var   = float(np.var(in_scores))
    out_var  = float(np.var(list(out_scores.values())))
    ratio    = in_var / out_var if out_var > 1e-10 else float("inf")
    in_sprd  = max(in_scores) - min(in_scores)
    out_sprd = max(out_scores.values()) - min(out_scores.values())

    print(f"\n  In-domain  spread={in_sprd:.4f}  variance={in_var:.6f}")
    print(f"  Out-domain spread={out_sprd:.4f}  variance={out_var:.6f}")
    print(f"  Variance ratio (in/out): {ratio:.2f}x")

    if pole_biased:
        print(f"  ALARM -- {len(pole_biased)} intruder(s) near poles:")
        for pb in pole_biased:
            print(f"    {pb['concept']:<24} {pb['score']:>+.4f} ({pb['pole']} pole)")

    gate_c_pass = (ratio >= 3.0) and (len(pole_biased) == 0)
    if gate_c_pass:
        print(f"  Gate C: PASS")
    elif pole_biased:
        print(f"  Gate C: FAIL (pole bias on {len(pole_biased)} intruder(s))")
    else:
        print(f"  Gate C: FAIL (variance ratio {ratio:.2f}x < 3.0x threshold)")

    return {
        "gate_c_pass":     gate_c_pass,
        "variance_ratio":  round(ratio, 3),
        "in_variance":     round(in_var, 6),
        "out_variance":    round(out_var, 6),
        "in_spread":       round(in_sprd, 4),
        "out_spread":      round(out_sprd, 4),
        "pole_biased":     pole_biased,
        "intruder_scores": out_scores,
        "n_intruders":     len(intruders),
    }

# -----------------------------------------------------------------------------
# CORE PROBE
# -----------------------------------------------------------------------------

def run_probe(domain_config, model, tokenizer, layer=24):
    name      = domain_config["name"]
    axis_desc = domain_config["axis_description"]
    pole_low  = domain_config["pole_low"]
    pole_high = domain_config["pole_high"]
    concepts  = domain_config["concepts"]
    gt_order  = domain_config["ground_truth_order"]
    null_low  = domain_config.get("null_pole_low",  "cloud")
    null_high = domain_config.get("null_pole_high", "democracy")
    intruders = domain_config.get("intruders", [])
    category  = domain_config.get("category", "uncategorized")
    key       = domain_config.get("key", name.lower().replace(" ", "_"))

    print(f"\n  [{category}] {name}")
    print(f"  {axis_desc}")

    t0 = time.time()

    axis      = get_axis_vector(pole_low,  pole_high,  model, tokenizer, layer)
    null_axis = get_axis_vector(null_low,  null_high,  model, tokenizer, layer)

    scores      = {}
    null_scores = {}
    for c in concepts:
        vec            = get_concept_vector(c, model, tokenizer, layer)
        scores[c]      = float(vec @ axis)
        null_scores[c] = float(vec @ null_axis)

    scored_concepts = [c for c in gt_order if c in scores]
    axis_vals       = [scores[c]      for c in scored_concepts]
    null_vals       = [null_scores[c] for c in scored_concepts]
    gt_ranks        = list(range(1, len(scored_concepts) + 1))

    rho,      p_rho  = spearmanr(axis_vals, gt_ranks)
    r,        p_r    = pearsonr( axis_vals, gt_ranks)
    null_rho, null_p = spearmanr(null_vals, gt_ranks)

    cat_min = min(axis_vals)
    cat_max = max(axis_vals)

    ranked      = sorted(scored_concepts, key=lambda c: scores[c])
    gt_rank_map = {c: i+1 for i, c in enumerate(scored_concepts)}
    ax_rank_map = {c: i+1 for i, c in enumerate(ranked)}

    print(f"  {'Ax':>3}  {'Concept':<22} {'Score':>8}  {'True':>5}  {'Err':>5}")
    print(f"  {'─'*50}")
    for ax_rank, concept in enumerate(ranked, 1):
        score = scores[concept]
        tr    = gt_rank_map.get(concept, "?")
        err   = ax_rank - tr if isinstance(tr, int) else None
        flag  = (" <<" if abs(err) >= 3 else " <" if abs(err) >= 2 else "") if err is not None else ""
        estr  = f"{err:+d}" if err is not None else "?"
        print(f"  {ax_rank:>3}  {concept:<22} {score:>+8.4f}  {str(tr):>5}  {estr:>5}{flag}")

    print(f"\n  rho={rho:+.4f} (p={p_rho:.3e})  null_rho={null_rho:+.4f}  delta={rho-null_rho:+.4f}")

    distortions = []
    for c in scored_concepts:
        err = ax_rank_map[c] - gt_rank_map[c]
        distortions.append({
            "concept":   c,
            "axis_rank": ax_rank_map[c],
            "true_rank": gt_rank_map[c],
            "error":     err,
        })
    distortions.sort(key=lambda x: abs(x["error"]), reverse=True)

    notable = [d for d in distortions if abs(d["error"]) >= 2]
    if notable:
        print(f"  Distortions (|err|>=2):")
        for d in notable:
            print(f"    {d['concept']:<22} err={d['error']:+d} "
                  f"({'HIGH' if d['error'] > 0 else 'LOW'})")

    sig        = p_rho < 0.05
    strong     = rho > 0.70
    beats_null = (rho - null_rho) > 0.30

    if sig and strong and beats_null:
        verdict = "CONFIRMED"
    elif sig and beats_null:
        verdict = "PARTIAL"
    elif sig:
        verdict = "WEAK"
    else:
        verdict = "NONE"

    intruder_result = None
    if intruders:
        intruder_result = run_intruder_check(
            intruders, axis, model, tokenizer, layer,
            axis_vals, cat_min, cat_max
        )

    elapsed   = time.time() - t0
    gc_status = ""
    if intruder_result is not None:
        gc_status = "  GateC=PASS" if intruder_result["gate_c_pass"] else "  GateC=FAIL"

    print(f"\n  VERDICT: {verdict}  ({elapsed:.1f}s){gc_status}")

    return {
        "key":          key,
        "domain":       name,
        "category":     category,
        "layer":        layer,
        "n_concepts":   len(scored_concepts),
        "spearman_rho": float(rho),
        "spearman_p":   float(p_rho),
        "pearson_r":    float(r),
        "pearson_p":    float(p_r),
        "null_rho":     float(null_rho),
        "null_p":       float(null_p),
        "delta_rho":    float(rho - null_rho),
        "verdict":      verdict,
        "gate_c":       intruder_result,
        "scores":       scores,
        "ranked_order": ranked,
        "distortions":  distortions,
        "elapsed_s":    round(elapsed, 1),
    }

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

def print_summary(results, output_txt="probe_summary.txt"):
    lines = []
    lines.append("=" * 76)
    lines.append("  RELATIONAL STRUCTURE PROBE -- SUMMARY")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(results)} domains")
    lines.append("=" * 76)

    confirmed = [r for r in results if r["verdict"] == "CONFIRMED"]
    partial   = [r for r in results if r["verdict"] == "PARTIAL"]
    weak      = [r for r in results if r["verdict"] == "WEAK"]
    none_     = [r for r in results if r["verdict"] == "NONE"]

    has_gc    = [r for r in results if r.get("gate_c") is not None]
    gc_pass   = [r for r in has_gc if r["gate_c"]["gate_c_pass"]]
    gc_fail   = [r for r in has_gc if not r["gate_c"]["gate_c_pass"]]

    lines.append(f"\n  Gate A+B: CONFIRMED={len(confirmed)}  PARTIAL={len(partial)}  "
                 f"WEAK={len(weak)}  NONE={len(none_)}")
    lines.append(f"  Confirmation rate: {100*len(confirmed)/max(len(results),1):.1f}%  "
                 f"(+PARTIAL: {100*(len(confirmed)+len(partial))/max(len(results),1):.1f}%)")

    if has_gc:
        lines.append(f"\n  Gate C (intruder): PASS={len(gc_pass)}/{len(has_gc)}  "
                     f"FAIL={len(gc_fail)}/{len(has_gc)}")
        if gc_fail:
            lines.append(f"  Failures:")
            for r in gc_fail:
                gc  = r["gate_c"]
                pb  = [x["concept"] for x in gc.get("pole_biased", [])]
                lines.append(f"    {r['domain'][:38]:<38} "
                             f"ratio={gc['variance_ratio']:.2f}x  "
                             f"biased={pb}")

    lines.append(f"\n{'─'*76}")
    lines.append(f"  {'Domain':<30} {'Cat':<12} {'rho':>7} {'p':>10} {'D-rho':>7}  V    GC")
    lines.append(f"{'─'*76}")

    for r in sorted(results, key=lambda x: x["spearman_rho"], reverse=True):
        sig  = "*" if r["spearman_p"] < 0.05 else " "
        gc   = r.get("gate_c")
        gstr = "P" if (gc and gc["gate_c_pass"]) else ("F" if gc else "-")
        lines.append(
            f"  {r['domain'][:29]:<30} {r['category'][:11]:<12} "
            f"{r['spearman_rho']:>+7.4f} {r['spearman_p']:>10.3e} "
            f"{r['delta_rho']:>+7.4f}  {r['verdict'][:4]}{sig}   {gstr}"
        )

    categories = sorted(set(r["category"] for r in results))
    lines.append(f"\n{'─'*76}")
    lines.append("  BY CATEGORY")
    lines.append(f"{'─'*76}")
    for cat in categories:
        cat_r    = [r for r in results if r["category"] == cat]
        mean_rho = np.mean([r["spearman_rho"] for r in cat_r])
        n_conf   = sum(1 for r in cat_r if r["verdict"] == "CONFIRMED")
        n_gc_p   = sum(1 for r in cat_r if r.get("gate_c") and r["gate_c"]["gate_c_pass"])
        n_gc_t   = sum(1 for r in cat_r if r.get("gate_c") is not None)
        gc_str   = f"GC={n_gc_p}/{n_gc_t}" if n_gc_t else "GC=n/a"
        lines.append(f"  {cat:<20} n={len(cat_r)}  mean_rho={mean_rho:+.4f}  "
                     f"confirmed={n_conf}/{len(cat_r)}  {gc_str}")

    if has_gc:
        lines.append(f"\n{'─'*76}")
        lines.append("  GATE C -- VARIANCE RATIOS")
        lines.append(f"  {'Domain':<34} {'ratio':>7}  {'in_sprd':>8}  "
                     f"{'out_sprd':>8}  {'result':>8}")
        lines.append(f"{'─'*76}")
        for r in sorted(has_gc, key=lambda x: x["gate_c"]["variance_ratio"],
                        reverse=True):
            gc  = r["gate_c"]
            pb  = len(gc.get("pole_biased", []))
            res = "PASS" if gc["gate_c_pass"] else f"FAIL(pb={pb})"
            lines.append(
                f"  {r['domain'][:33]:<34} {gc['variance_ratio']:>7.2f}x "
                f"{gc['in_spread']:>8.4f}  {gc['out_spread']:>8.4f}  {res}"
            )

    lines.append(f"\n{'─'*76}")
    lines.append("  TOP DISTORTIONS (|error| >= 4)")
    lines.append(f"{'─'*76}")
    all_dist = []
    for r in results:
        for d in r.get("distortions", []):
            if abs(d["error"]) >= 4:
                all_dist.append({**d, "domain": r["domain"]})
    all_dist.sort(key=lambda x: abs(x["error"]), reverse=True)
    for d in all_dist[:40]:
        direction = "HIGH" if d["error"] > 0 else "LOW "
        lines.append(f"  {d['concept']:<22} err={d['error']:+d}  "
                     f"{direction}  [{d['domain']}]")

    all_biased = []
    for r in results:
        gc = r.get("gate_c")
        if gc:
            for pb in gc.get("pole_biased", []):
                all_biased.append({**pb, "domain": r["domain"]})
    if all_biased:
        lines.append(f"\n{'─'*76}")
        lines.append("  ALARM: POLE-BIASED INTRUDERS")
        lines.append(f"  Out-of-domain concepts that landed near a pole.")
        lines.append(f"  These suggest the axis measures something broader than its label.")
        lines.append(f"{'─'*76}")
        for pb in sorted(all_biased, key=lambda x: abs(x["score"]), reverse=True):
            lines.append(
                f"  {pb['concept']:<22} score={pb['score']:>+8.4f}  "
                f"{pb['pole']:<4} pole  [{pb['domain']}]"
            )

    text = "\n".join(lines)
    print("\n" + text)
    with open(output_txt, "w") as f:
        f.write(text)
    print(f"\nSummary saved to: {output_txt}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains",  default="domains.json")
    parser.add_argument("--output",   default="probe_results.json")
    parser.add_argument("--summary",  default="probe_summary.txt")
    parser.add_argument("--layer",    type=int, default=24)
    parser.add_argument("--category", nargs="+", default=None)
    parser.add_argument("--keys",     nargs="+", default=None)
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--list",     action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.domains):
        print(f"ERROR: {args.domains} not found.")
        sys.exit(1)

    with open(args.domains) as f:
        all_domains = json.load(f)

    if args.list:
        by_cat = {}
        for d in all_domains:
            by_cat.setdefault(d.get("category", "?"), []).append(d)
        print(f"\n{len(all_domains)} domains in {args.domains}:\n")
        for cat, doms in sorted(by_cat.items()):
            ni = sum(1 for d in doms if d.get("intruders"))
            print(f"  [{cat}]  ({len(doms)} domains, {ni} with intruder tests)")
            for d in doms:
                n = len(d.get("intruders", []))
                flag = f"  [{n} intruders]" if n else ""
                print(f"    {d['key']:<38} {d['name']}{flag}")
        sys.exit(0)

    domains_to_run = list(all_domains)
    if args.keys:
        domains_to_run = [d for d in domains_to_run if d.get("key") in args.keys]
    if args.category:
        domains_to_run = [d for d in domains_to_run
                          if d.get("category") in args.category]

    existing_results = []
    completed_keys   = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing_results = json.load(f)
        completed_keys = {r["key"] for r in existing_results}
        domains_to_run = [d for d in domains_to_run
                          if d.get("key") not in completed_keys]
        print(f"Resuming: {len(completed_keys)} done, "
              f"{len(domains_to_run)} remaining.")

    if not domains_to_run:
        print("Nothing to run. Use --list to see domains.")
        sys.exit(0)

    print(f"Running {len(domains_to_run)} domains | layer={args.layer}")
    print("Loading model...")
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    print("Model loaded.\n")

    # =========================================================================
    # ENFORCE VB_CORE VALIDATION GATES
    # =========================================================================
    print("Executing VectorBible Foundation Validation...")
    
    g1_pass = gate_1_extractor_sanity(model, tokenizer, args.layer)
    if not g1_pass:
        print("\nFATAL ERROR: Gate 1 (Extractor Sanity) failed.")
        print("Nothing above this gate is trustworthy. Aborting execution.")
        sys.exit(1)

    g2_pass, real_rho = gate_2_geographic_ordering(model, tokenizer, args.layer)
    if not g2_pass:
        print("\nFATAL ERROR: Gate 2 (Geographic Ordering) failed.")
        print("Nothing above this gate is trustworthy. Aborting execution.")
        sys.exit(1)

    g3_pass = gate_3_null_model(model, tokenizer, args.layer, real_rho)
    if not g3_pass:
        print("\nFATAL ERROR: Gate 3 (Null Model Control) failed.")
        print("Nothing above this gate is trustworthy. Aborting execution.")
        sys.exit(1)

    print("\n" + "="*60)
    print("ALL FOUNDATION GATES PASSED. Proceeding to domain probes.")
    print("="*60 + "\n")
    # =========================================================================

    results = list(existing_results)
    t_start = time.time()
    
    
    for i, cfg in enumerate(domains_to_run, 1):
        print(f"\n{'#'*60}")
        print(f"  {i}/{len(domains_to_run)}")
        try:
            r = run_probe(cfg, model, tokenizer, args.layer)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "key":     cfg.get("key", "?"),
                "domain":  cfg.get("name", "?"),
                "category": cfg.get("category", "?"),
                "error":   str(e),
                "verdict": "ERROR",
                "spearman_rho": 0,
                "delta_rho": 0,
                "gate_c": None,
            })

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = time.time() - t_start
        eta     = (elapsed / i) * (len(domains_to_run) - i)
        print(f"  Saved. Elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    valid = [r for r in results if r.get("verdict") != "ERROR"]
    print_summary(valid, args.summary)
    print(f"\nDone. {len(valid)}/{len(results)} succeeded.")