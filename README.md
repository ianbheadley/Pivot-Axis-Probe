# Relational Structure Probe for LLMs

## Project Overview

This project investigates the internal representations of Large Language Models (LLMs) by probing their understanding of various relational, hierarchical, and temporal structures. The primary technique used is an **axis probe technique**, which extracts conceptual axes from the model's activation space and evaluates how well the model organizes concepts along these axes.

Specifically, the probe defines a domain (e.g., "Biological Size" or "Historical Temporal Order") by specifying two pole concepts (e.g., "virus" and "elephant" for size, or "Stone Age" and "internet" for time). The probe then extracts an axis vector between these two poles in the activation space of the LLM (in this case, Meta-Llama-3-8B-Instruct-4bit).

The model's understanding is evaluated by projecting various in-domain concepts onto this axis and ranking them based on their projection scores. The performance is assessed using three main validation gates:
1. **Gate A (Spearman Rank Correlation):** Checks if the model's ranking of concepts along the axis significantly correlates with the ground truth order.
2. **Gate B (Null Model Control):** Compares the target axis against a control "null" axis to ensure the ordering is specific to the semantic domain and not a general artifact of the model's activation space.
3. **Gate C (Intruder Validation):** Projects out-of-domain "intruder" concepts onto the axis to check for variance and pole bias. A successful axis should tightly cluster unrelated concepts near the center (neutral zone) while spreading in-domain concepts across the axis.

## Test Results Findings

A total of 53 diverse domains were probed across categories such as physical, biological, cognitive, temporal, geographic, psychological, social, technology, and aesthetic. The results indicate a robust capability of the model to organize concepts structurally.

### High-Level Summary
* **Overall Confirmation:** 48 out of 53 domains (90.6%) were strictly "CONFIRMED" (passing Gates A and B with high significance and delta). Including partial successes, the rate rises to 92.5%.
* **Intruder Robustness (Gate C):** 49 out of 53 domains (92.4%) passed the intruder validation check, demonstrating that the axes are generally well-defined and exclusive to their target domains.
* **Top Performing Domains:**
  * Cognitive Complexity (Spearman rho = +0.9879)
  * Relative Brain Size (rho = +0.9780)
  * Data Storage Capacity (rho = +0.9762)
  * Sleep / Consciousness Depth (rho = +0.9762)
  * Economic Wealth (rho = +0.9758)
* **Categories:** The psychological (mean rho = +0.9607), biological (+0.9479), and social (+0.9224) categories demonstrated the highest average correlations.

### Notable Distortions and Failures
While the model performs exceptionally well, there are notable distortions where concepts are placed out of order. For example:
* In the **US Cities North-South** domain, *New York* was ranked significantly lower (more southern) than its true geographic position.
* In **Biological Size**, *human* was ranked smaller than its true relative size compared to other animals.
* In **Disease Severity Progression**, the concept *exposed* was ranked much higher in severity than it should be.

Four domains failed the Gate C intruder test due to pole bias, where an out-of-domain concept strongly aligned with one of the poles. For instance, the concept *whale* strongly aligned with the high pole of the **Food Chain / Trophic Level** axis, suggesting the axis might be conflating "trophic level" with "physical size" or "oceanic dominance". Similarly, *homeless* aligned with the low pole of the **Academic Credential** axis, indicating a conflation between academic rank and socioeconomic status.

## Relation to LLM Research & Implications

The findings from this probe have significant implications for our understanding of Large Language Models:

1. **Structured Internal Representations:** The high confirmation rate across 53 diverse domains provides strong evidence that LLMs do not merely memorize statistical co-occurrences of tokens. Instead, they construct structured, continuous, and multi-dimensional internal representations of the world. Concepts are organized geographically, temporally, physically, and conceptually in a manner that aligns with human intuition and ground truth.
2. **Axis Extraction as an Interpretability Tool:** The success of the axis probe technique demonstrates that linear directions in the activation space carry specific semantic meanings. This supports the linear representation hypothesis in LLMs, suggesting that complex concepts can be decomposed into linear combinations of simpler, interpretable features (axes).
3. **Semantic Conflation and Bias:** The intruder alarms (Gate C failures) and specific rank distortions highlight the model's limitations and potential biases. When *homeless* aligns with the low pole of an *Academic Credential* axis, it reveals that the model's internal representation of these concepts is entangled. This kind of semantic conflation can lead to downstream biases and hallucinations if the model relies on these entangled axes for reasoning.

## Future Research Directions

Based on these findings, several promising directions for future research emerge:

1. **Cross-Model and Scale Comparisons:** Replicate the probe across different model families (e.g., Mistral, Gemma, Qwen) and different parameter scales (e.g., 8B vs. 70B vs. 405B) to investigate how these relational structures evolve with scale and training data.
2. **Layer-wise Analysis:** Conduct a deep layer-by-layer analysis. Currently, the probe evaluates layer 24. Tracing how these axes form and dissolve across the early, middle, and late layers of the model could reveal the exact mechanisms of concept formation.
3. **Causal Interventions:** Move beyond probing (observation) to causal interventions. If a specific axis controls the "size" of an object, can we artificially manipulate a concept's projection on this axis to change the model's downstream generative output (e.g., making the model describe an ant as massive)?
4. **Expanding Semantic Domains:** Test more abstract, culturally subjective, or highly specialized domains to find the limits of the model's structured understanding.
5. **Mitigating Semantic Entanglement:** Develop fine-tuning or activation-steering techniques aimed at disentangling conflated axes (e.g., separating "academic credential" from "socioeconomic status") to reduce model bias.