"""
vb_core.py — VectorBible Foundation Layer
==========================================
The single file everything else builds on.

Before running any experiment, this file must pass all three
validation gates. If any gate fails, nothing above it is trustworthy.

GATE 1: Extractor sanity     — norms uniform, semantics make sense
GATE 2: Geometric ordering   — south→north axis works on 5 cities
GATE 3: Null model           — random axis produces near-zero correlation

Usage:
    python vb_core.py                  # runs all three gates
    python vb_core.py --layer 20       # test a different layer
    python vb_core.py --validate-only  # gates only, no interactive mode
"""

import argparse
import numpy as np
import mlx.core as mx
from mlx_lm import load
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# THE EXTRACTOR
# This is the only function that touches the model internals.
# Every other tool calls this. Get this right and everything else is buildable.
# ─────────────────────────────────────────────────────────────────────────────

def get_concept_vector(
    concept: str,
    model,
    tokenizer,
    layer: int = 20,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract a single concept's representation from a specific transformer layer.

    Design decisions — each one matters:

    1. CONCEPT-POSITION MEAN POOLING, not last-token of a template.
       Previous approach: "The concept known as {concept} is" → take last token.
       Problem: the last token is always "is" — same grammatical slot for all
       concepts, heavily shaped by template context, not the concept itself.
       dog/entropy came out at cosine=0.71 because "is" is "is" regardless
       of what precedes it.

       Fix: use a short prefix template "Concept: {concept}", then identify
       exactly which token positions correspond to the concept itself, and
       mean-pool the hidden states at those positions only. The template
       provides attention context but its positions are discarded.

    2. TOKEN SPAN DETECTION via prefix-length comparison.
       Tokenize the prefix alone to know where it ends. The concept tokens
       are everything after that prefix in the full sequence. This works
       reliably across all tokenizers regardless of BOS handling.

    3. L2 NORMALIZATION before returning.
       Raw hidden state norms vary by token frequency and layer depth.
       Normalization ensures downstream dot products = cosine similarity.

    4. LAYER CHOICE default = 20.
       Layers 18-24 are where abstract semantic geometry is most stable
       in Llama-3-8B. Layer choice is a hyperparameter — validated in gates.

    Args:
        concept:    The concept to embed (e.g. "Paris", "entropy", "Seattle")
        model:      Loaded MLX model
        tokenizer:  Loaded MLX tokenizer
        layer:      Which transformer layer to extract from (0-31)
        normalize:  Whether to L2-normalize the output vector

    Returns:
        np.ndarray of shape (4096,) — the concept's representation
    """
    prefix = "Concept: "
    full_prompt = prefix + concept

    # Tokenize both to find exact concept token span
    prefix_tokens = tokenizer.encode(prefix)
    full_tokens   = tokenizer.encode(full_prompt)

    n_prefix  = len(prefix_tokens)
    n_full    = len(full_tokens)
    n_concept = n_full - n_prefix  # number of tokens the concept takes

    # Safety: if span detection fails, fall back to last token
    if n_concept <= 0:
        n_concept = 1

    x = mx.array([full_tokens])
    seq_len = x.shape[1]

    # Causal attention mask
    mask = mx.triu(
        mx.full((seq_len, seq_len), -mx.inf, dtype=mx.float16), k=1
    )

    # Forward pass through specified layer
    v = model.model.embed_tokens(x)
    for i in range(layer):
        v = model.model.layers[i](v, mask=mask)

    mx.eval(v)

    # Extract hidden states at concept token positions and mean pool
    # Shape: (seq_len, 4096) → slice concept span → mean → (4096,)
    hidden_all = np.array(v[0], dtype=np.float32)
    concept_hidden = hidden_all[-n_concept:]        # last n_concept positions
    pooled = concept_hidden.mean(axis=0)

    if normalize:
        norm = np.linalg.norm(pooled)
        pooled = pooled / (norm + 1e-8)

    return pooled


def get_axis_vector(
    pole_low: str,
    pole_high: str,
    model,
    tokenizer,
    layer: int = 20,
    n_anchors: int = 1
) -> np.ndarray:
    """
    Build a directional axis vector from two semantic poles.

    For a single anchor pair: axis = normalize(high - low)
    For n_anchors > 1: axis = normalize(mean(highs) - mean(lows))
    Multiple anchors produce a more stable axis — less sensitive
    to quirks of any single concept's tokenization.

    The axis is L2-normalized so downstream dot products give
    scalar projections on a consistent scale.
    """
    if isinstance(pole_low, str):
        pole_low = [pole_low]
    if isinstance(pole_high, str):
        pole_high = [pole_high]

    v_lows  = [get_concept_vector(p, model, tokenizer, layer) for p in pole_low]
    v_highs = [get_concept_vector(p, model, tokenizer, layer) for p in pole_high]

    mean_low  = np.mean(v_lows,  axis=0)
    mean_high = np.mean(v_highs, axis=0)

    axis = mean_high - mean_low
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return axis


def project_concepts(
    concepts: list,
    axis: np.ndarray,
    model,
    tokenizer,
    layer: int = 20
) -> dict:
    """
    Project a list of concepts onto an axis.
    Returns a dict: concept -> scalar score, sorted low to high.
    """
    scores = {}
    for c in concepts:
        vec = get_concept_vector(c, model, tokenizer, layer)
        scores[c] = float(vec @ axis)
    return dict(sorted(scores.items(), key=lambda x: x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION GATES
# All three must pass before any experiment is run.
# ─────────────────────────────────────────────────────────────────────────────

def gate_1_extractor_sanity(model, tokenizer, layer: int) -> bool:
    """
    GATE 1: Extractor Sanity
    
    Tests:
      A. Norm uniformity — all concept vectors should have norm ≈ 1.0
         (proves normalization is working and there's no bimodal split)
      
      B. Semantic ordering — similar concepts should have higher cosine
         similarity than dissimilar concepts
         (proves the vectors carry semantic content, not just token artifacts)
    
    Pass criteria:
      A. All norms between 0.99 and 1.01
      B. All 3 similar pairs outscore all 2 dissimilar pairs
    """
    print("\n" + "="*60)
    print("GATE 1: Extractor Sanity")
    print("="*60)

    # Test A: Norm uniformity
    # Include multi-token concepts specifically — these were the broken ones
    concepts = [
        "Paris",        # 1 token
        "Tokyo",        # 2 tokens
        "democracy",    # 3 tokens
        "Shakespeare",  # 4 tokens
        "iPhone",       # 2 tokens
        "entropy",      # 3 tokens
    ]

    print("\nA. Norm Uniformity (all should be 1.000 ± 0.001):")
    norms = []
    for c in concepts:
        vec = get_concept_vector(c, model, tokenizer, layer)
        n = float(np.linalg.norm(vec))
        norms.append(n)
        status = "✓" if 0.999 <= n <= 1.001 else "✗ FAIL"
        n_tokens = len(tokenizer.encode(c)) - 1  # subtract BOS
        print(f"  {c:<16} norm={n:.6f}  tokens={n_tokens}  {status}")

    norm_pass = all(0.999 <= n <= 1.001 for n in norms)
    print(f"  Result: {'PASS' if norm_pass else 'FAIL'}")

    # Test B: Semantic ordering
    # Similar pairs: genuine semantic neighbors — high cosine expected (>0.6)
    # Dissimilar pairs: unrelated domains — low cosine expected (<0.3)
    # Note: Paris/France is NOT a good similar pair — city vs country,
    # related but not semantic neighbors. Using cleaner pairs:
    similar_pairs = [
        ("dog",    "puppy"),    # near-synonyms, same domain
        ("king",   "queen"),    # semantic neighbors, same domain
        ("hot",    "warm"),     # scalar neighbors
    ]
    dissimilar_pairs = [
        ("dog",    "algorithm"),
        ("king",   "photosynthesis"),
    ]

    print("\nB. Semantic Ordering:")
    print("  Similar pairs (cosine sim should be > dissimilar pairs):")
    sim_scores = []
    for a, b in similar_pairs:
        va = get_concept_vector(a, model, tokenizer, layer)
        vb = get_concept_vector(b, model, tokenizer, layer)
        cos = float(va @ vb)
        sim_scores.append(cos)
        print(f"  {a}/{b}: {cos:.4f}")

    print("  Dissimilar pairs:")
    dissim_scores = []
    for a, b in dissimilar_pairs:
        va = get_concept_vector(a, model, tokenizer, layer)
        vb = get_concept_vector(b, model, tokenizer, layer)
        cos = float(va @ vb)
        dissim_scores.append(cos)
        print(f"  {a}/{b}: {cos:.4f}")

    semantic_pass = min(sim_scores) > max(dissim_scores)
    print(f"\n  Min similar: {min(sim_scores):.4f}  "
          f"Max dissimilar: {max(dissim_scores):.4f}")
    print(f"  Result: {'PASS — similar pairs consistently outscore dissimilar' if semantic_pass else 'FAIL — overlap between similar and dissimilar scores'}")
    print(f"  Note: dog/algorithm and king/photosynthesis should be well below 0.5")
    print(f"        dog/puppy and king/queen should be well above 0.6")

    gate_pass = norm_pass and semantic_pass
    print(f"\nGATE 1: {'✓ PASS' if gate_pass else '✗ FAIL — fix extractor before proceeding'}")
    return gate_pass


def gate_2_geographic_ordering(model, tokenizer, layer: int) -> bool:
    """
    GATE 2: Geographic Ordering
    
    Tests whether the city-anchored South→North axis correctly orders
    5 cities by latitude without any labels.
    
    This is the minimum viable version of Experiment 2.
    
    Pass criteria:
      Spearman ρ > 0.70 (strong monotonic ordering)
      p < 0.10 (significant given only 5 data points)
      
    Note on p-value: with n=5, even ρ=1.0 gives p=0.017.
    ρ=0.90 gives p≈0.04. So p<0.10 is the right threshold here.
    """
    print("\n" + "="*60)
    print("GATE 2: Geographic Ordering")
    print("="*60)

    # Multi-anchor axis: mean of 2 southern cities minus mean of 2 northern
    # More stable than a single pair
    print("\nBuilding South→North axis from city anchors...")
    print("  South anchors: Miami Florida, Houston Texas")
    print("  North anchors: Anchorage Alaska, Minneapolis Minnesota")

    axis = get_axis_vector(
        pole_low  = ["Miami Florida",    "Houston Texas"],
        pole_high = ["Anchorage Alaska", "Minneapolis Minnesota"],
        model=model, tokenizer=tokenizer, layer=layer
    )

    # Test cities — not used in axis construction
    test_cities = ["Miami",      "Atlanta",   "Chicago",  "Seattle",  "Anchorage"]
    true_lats   = [ 25.8,         33.7,         41.9,       47.6,       61.2]

    print("\nProjecting test cities onto axis:")
    scores = []
    for city, lat in zip(test_cities, true_lats):
        vec = get_concept_vector(city, model, tokenizer, layer)
        score = float(vec @ axis)
        scores.append(score)
        print(f"  {city:<12} score={score:+.4f}  true_lat={lat}°N")

    rho, p = spearmanr(scores, true_lats)
    print(f"\nSpearman ρ = {rho:.4f}  (p = {p:.4f})")

    # Show what the axis thinks the order is
    ranked = sorted(zip(test_cities, scores), key=lambda x: x[1])
    print(f"Axis-implied order (S→N): {' → '.join(c for c,_ in ranked)}")
    print(f"True order       (S→N): {' → '.join(c for c,_ in sorted(zip(test_cities, true_lats), key=lambda x: x[1]))}")

    gate_pass = rho > 0.70 and p < 0.10
    print(f"\nGATE 2: {'✓ PASS' if gate_pass else '✗ FAIL — try layer 24, or check anchor concepts'}")
    return gate_pass, rho


def gate_3_null_model(model, tokenizer, layer: int, real_rho: float) -> bool:
    """
    GATE 3: Null Model Comparison
    
    Tests whether the geographic signal is specific to the geographic axis
    or whether ANY axis produces correlation with latitude.
    
    A random/unrelated axis should produce ρ ≈ 0 (within noise).
    The real axis should produce substantially higher ρ than the null.
    
    Pass criteria:
      |null_rho| < 0.40
      real_rho - null_rho > 0.30  (meaningful signal over noise)
    
    This is the control that was missing from all previous runs.
    Without it, a positive ρ proves nothing.
    """
    print("\n" + "="*60)
    print("GATE 3: Null Model Control")
    print("="*60)

    # Null axis: semantically unrelated to geography
    print("\nBuilding null axis (Apple → Philosophy)...")
    null_axis = get_axis_vector(
        pole_low  = "Apple",
        pole_high = "Philosophy",
        model=model, tokenizer=tokenizer, layer=layer
    )

    test_cities = ["Miami",      "Atlanta",  "Chicago",  "Seattle",  "Anchorage"]
    true_lats   = [ 25.8,         33.7,       41.9,       47.6,       61.2]

    print("Projecting same cities onto null axis:")
    null_scores = []
    for city, lat in zip(test_cities, true_lats):
        vec = get_concept_vector(city, model, tokenizer, layer)
        score = float(vec @ null_axis)
        null_scores.append(score)
        print(f"  {city:<12} null_score={score:+.4f}  true_lat={lat}°N")

    null_rho, null_p = spearmanr(null_scores, true_lats)

    print(f"\nNull axis:  ρ = {null_rho:+.4f}  (p = {null_p:.4f})")
    print(f"Real axis:  ρ = {real_rho:+.4f}")
    print(f"Signal gap: Δρ = {real_rho - null_rho:+.4f}")

    gate_pass = abs(null_rho) < 0.40 and (real_rho - null_rho) > 0.30
    print(f"\nGATE 3: {'✓ PASS — geographic signal is specific, not generic' if gate_pass else '✗ FAIL — signal not significantly above null'}")
    return gate_pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(layer: int = 20):
    """
    Runs all three gates in order.
    Stops and reports clearly if any gate fails.
    """
    print("\n" + "█"*60)
    print("  VectorBible Foundation Validation")
    print(f"  Model: mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    print(f"  Layer: {layer}")
    print("█"*60)

    print("\nLoading model...")
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    print("Model loaded.")

    # Gate 1
    g1 = gate_1_extractor_sanity(model, tokenizer, layer)
    if not g1:
        print("\n" + "!"*60)
        print("STOPPED AT GATE 1.")
        print("Fix _get_hidden_state before running Gate 2 or 3.")
        print("!"*60)
        return False

    # Gate 2
    g2, real_rho = gate_2_geographic_ordering(model, tokenizer, layer)
    if not g2:
        print("\n" + "!"*60)
        print("STOPPED AT GATE 2.")
        print("Geographic axis failed. Try --layer 24.")
        print("The extractor works but the axis construction needs tuning.")
        print("!"*60)
        return False

    # Gate 3
    g3 = gate_3_null_model(model, tokenizer, layer, real_rho)

    # Final report
    print("\n" + "█"*60)
    print("  VALIDATION SUMMARY")
    print("█"*60)
    print(f"  Gate 1 — Extractor sanity:   {'✓ PASS' if g1 else '✗ FAIL'}")
    print(f"  Gate 2 — Geographic ordering: {'✓ PASS' if g2 else '✗ FAIL'}")
    print(f"  Gate 3 — Null model control:  {'✓ PASS' if g3 else '✗ FAIL'}")

    all_pass = g1 and g2 and g3
    print("\n" + ("█"*60))
    if all_pass:
        print("  ALL GATES PASS.")
        print("  Foundation is verified. Safe to build experiments on top.")
        print("  Import get_concept_vector and get_axis_vector from this file.")
    else:
        print("  ONE OR MORE GATES FAILED.")
        print("  Do not proceed to experiments until all gates pass.")
    print("█"*60 + "\n")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=20,
                        help="Transformer layer to extract from (default: 20)")
    args = parser.parse_args()
    run_validation(layer=args.layer)