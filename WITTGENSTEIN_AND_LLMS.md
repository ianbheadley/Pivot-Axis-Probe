# Wittgenstein and the Relational Structure of LLMs

The success of the axis probe technique in revealing structured, continuous internal representations within Large Language Models (LLMs) invites a profound philosophical comparison. How does a model trained merely to predict the next token develop such rich, organized mappings of concepts like biological size, geographical location, or temporal order?

The philosophical frameworks developed by Ludwig Wittgenstein offer a compelling lens through which to interpret these findings, bridging the gap between statistical machine learning and human semantics.

## 1. The Later Wittgenstein: Meaning as Use

In his later work, particularly *Philosophical Investigations*, Wittgenstein argued against the idea that words have fixed, essential meanings that point to objects in the world. Instead, he proposed that **"meaning is use"**—the meaning of a word is defined by how it is used within the context of a language.

### Connection to LLMs
LLMs are the ultimate empiricists of language use. They have no physical embodiment or direct experience of the world. Everything an LLM "knows" about an elephant or a virus is derived entirely from how those tokens co-occur with other tokens in vast corpora of human text.

The findings from this project—that an LLM organizes concepts along a continuous axis (e.g., from small to large) with a high degree of correlation (+0.9479 for biology)—demonstrate that *statistical patterns of use* are sufficient to reconstruct complex relational structures. The model learns that "elephant" is used in contexts involving "heavy," "large," and "stomps," while "virus" is used near "microscopic," "invisible," and "infects." By predicting token sequences based on these usage patterns, the LLM constructs an internal geometry where "meaning as use" crystallizes into measurable, linear distances in activation space.

## 2. Family Resemblance and Continuous Geometry

Wittgenstein also introduced the concept of **"family resemblance"** to explain how we categorize things. He argued that things we call "games" (board games, card games, Olympic games) do not share one single essential feature. Instead, they share a complex, overlapping network of similarities—like the overlapping physical traits of family members.

### Connection to LLMs
The continuous nature of the internal representations probed in this project strongly supports a "family resemblance" model of categorization rather than a rigid, discrete one. The axis probe does not find a hard boundary between "small" and "large"; it finds a continuous gradient.

Furthermore, the "intruder robustness" failures (e.g., *whale* aligning with the high pole of the *Food Chain* axis, or *homeless* aligning with the low pole of *Academic Credential*) highlight how concepts cluster based on complex, overlapping semantic features. The model's representation of "whale" shares statistical "family resemblances" with concepts of physical dominance and ocean hierarchy, which occasionally conflate with pure trophic level. These semantic entanglements in the vector space mirror the fuzzy, overlapping boundaries of human concepts that Wittgenstein described.

## 3. The Early Wittgenstein: The Picture Theory of Language

While the later Wittgenstein dominates discussions of LLMs, his earlier work, the *Tractatus Logico-Philosophicus*, is also surprisingly relevant. In the *Tractatus*, Wittgenstein proposed the **"picture theory of language,"** suggesting that language is a logical picture of the facts in the world. The structure of a meaningful proposition mirrors the logical structure of reality.

### Connection to LLMs
The high confirmation rate (90.6%) across 53 diverse domains (physical, cognitive, temporal, etc.) suggests that the LLM is doing more than just memorizing text—it is building an internal "picture" or model of the world's structure.

The fact that we can extract a North-South geographical axis or a Historical Temporal Order axis demonstrates that the model's internal activation space is topologically isomorphic (structurally similar) to the real-world phenomena those words describe. The linear directions in the activation space are the "logical pictures" of the structural relationships in the physical and conceptual world.

---

## Future Directions for LLM Interpretability Tools

The current axis probe technique is a powerful tool, but it is just the beginning. By building on these philosophical insights and our growing understanding of activation spaces, we can develop more advanced tools for LLM interpretability.

### 1. Multi-Dimensional Manifold Probes
**Concept:** Currently, we extract a 1D linear axis between two poles. However, many human concepts (and likely LLM representations) are not strictly linear. Wittgenstein's family resemblances suggest complex, multi-dimensional clusters.
**Tool:** Develop probes that extract 2D planes, 3D volumes, or higher-dimensional manifolds for complex domains. For example, instead of a single "Political Left-Right" axis, a tool could extract a 2D plane capturing "Economic Left-Right" and "Authoritarian-Libertarian" simultaneously, allowing us to map concepts in a richer semantic space.

### 2. Contextual "Language Game" Probes
**Concept:** Wittgenstein argued that meaning depends on the specific "language game" being played (e.g., scientific description vs. poetic metaphor). An LLM's activation space likely shifts depending on the prompted context.
**Tool:** Create probes that evaluate how conceptual axes shift or warp under different contexts or personas.
*   *Experiment:* Probe the "Size" axis with the standard prompt, and then probe the exact same concepts but prepend the prompt with "You are writing a surrealist poem..." or "You are a quantum physicist...". The tool would measure the divergence or rotation of the axis across different language games.

### 3. Causal Intervention and Axis Editing
**Concept:** Probing is observational. To truly understand if the model *relies* on these structures for reasoning, we need to intervene.
**Tool:** Develop causal intervention tools that allow us to project a concept's activation vector onto a specific axis and manually shift it.
*   *Experiment:* If we find the activation for "Apple" and artificially slide it up the "Biological Size" axis (towards "Elephant"), does the LLM suddenly start generating text describing giant, house-sized apples? This would prove that the structural axis is causally responsible for the model's output behavior, not just a byproduct.

### 4. Entanglement and Conflation Analyzers (Bias Detectors)
**Concept:** The Gate C intruder failures show that models conflate distinct axes (e.g., Socioeconomic Status vs. Academic Credentials).
**Tool:** Build tools specifically designed to find orthogonal axes that *should* be independent but are actually correlated in the model's space. By automatically scanning thousands of paired axes, we can generate a "Conflation Map" to identify latent biases and semantic entanglements before a model is deployed. This acts as an automated, structural bias detector.