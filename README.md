# Relational Structure Probe for LLMs

  

## Project Overview

This project investigates the internal representations of Large Language Models (LLMs) by probing their understanding of various relational, hierarchical, and temporal structures. The primary technique used is an **axis probe technique**, which extracts conceptual axes from the model's activation space and evaluates how well terms between two extremes map between them.

Specifically, the probe defines a domain (e.g., "Biological Size" or "Historical Temporal Order") by specifying extreme and opposite pole concepts (e.g., "virus" and "elephant" for size, or "Stone Age" and "internet" for time). The probe then extracts an axis vector between these two poles in the activation space of the LLM (in this case, Meta-Llama-3-8B-Instruct-4bit).
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
