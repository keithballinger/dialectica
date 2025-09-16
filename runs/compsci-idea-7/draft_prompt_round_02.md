You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, except for anything blockchain related
- Highly novel
- Publishable in a leading journal for its subfield

        Draft:
        Major Revisions

Title
A Percolation Threshold Governs In-Context Learning in Transformers

Abstract
We propose a structural theory of in-context learning (ICL) in Transformers: strong ICL emerges when the per-layer attention graph crosses a percolation threshold that guarantees low-stretch paths from prompt tokens to targets within the model’s depth. We formalize the attention computation as a directed per-layer graph and identify a reproduction number z that controls a directed percolation transition. When z ≤ 1, directed paths from prompts to targets are exponentially rare with depth, bounding the mutual information that predictions can extract from prompts. When z > 1 and the induced graph has small chemical distances (shortest-path lengths) between typical prompt–target pairs, the network admits exponentially many edge-disjoint, low-stretch routes, enabling robust ICL. The theory yields falsifiable predictions: threshold-like ICL gains as z crosses 1; depth–connectivity tradeoffs governed by the graph’s diameter; and a collapse of ICL performance when per-layer paths are forced to exceed depth. We outline an experimental program using sparse-attention Transformers with controlled connectivity to test for phase-transition-like behavior in ICL performance precisely at the predicted percolation regime.

Introduction
Transformer models often learn tasks from prompts without parameter updates, a phenomenon termed in-context learning (ICL). Despite empirical progress, the structural conditions that enable ICL remain unclear. Prior accounts emphasize algorithmic specialization, meta-learning dynamics, or retrieval-and-induction circuits, but do not predict when ICL should suddenly emerge or collapse as architectural sparsity, head count, window size, or depth vary.

We propose a graph-theoretic account: attention induces a directed, per-layer communication graph over sequence positions. Across depth, information propagates along directed sequences of attention edges. ICL requires that examples in the prompt reach the prediction site within the layer budget, and that the paths used are “low-stretch” in the underlying graph so that signals are not forced through long detours that exceed depth or diffuse attention mass.

Our core claims are:
- Percolation control: Define z as the expected number of supra-threshold outgoing connections per token per layer (accounting for heads and sparsity). A directed percolation transition at z = 1 governs the existence of macroscopic, depth-spanning paths from prompts to targets.
- Low-stretch requirement: Effective ICL requires typical prompt–target graph distances to be at most the model depth L. Sparse patterns that keep z > 1 but inflate chemical distances beyond L suppress ICL.
- Predictive, falsifiable thresholds: Families of sparse-attention Transformers should exhibit phase-transition-like ICL gains precisely when z crosses 1 and typical distances fall below L.

This view unifies diverse observations—emergent abilities with scale, sharp gains from added heads or wider windows, and sensitivity to sparsity masks—under a common structural criterion.

Method
Formalizing attention as a per-layer directed graph
- Sequence and architecture: Consider a sequence of n tokens and a Transformer with L layers and H heads per layer. For layer l and head h, let a_l,h(j ← i) be the attention weight from source token i at layer l−1 to target token j at layer l.
- Edge set: Fix a transmission threshold τ ∈ (0, 1). We place a directed edge i → j in layer l if there exists a head h with a_l,h(j ← i) ≥ τ. Let G_l(τ) denote the resulting directed graph from layer l−1 to layer l. Multi-head edges are unioned. Residual connections provide i → i always.
- Reproduction number: Let z_l(τ) be the expected out-degree of a node in G_l(τ), equivalently the expected number of targets j per source i that receive at least τ mass from any head. Let z(τ) be the typical per-layer reproduction number (e.g., the geometric mean across layers). Intuitively, z captures how many downstream positions a token can reliably influence per layer.

Directed percolation and depth-limited reachability
- Per-layer percolation: In directed random graphs, a giant (in- and out-) component appears at mean degree > 1. z > 1 implies that, for large n, a positive fraction of nodes have macroscopic reachability sets across layers; z ≤ 1 implies exponentially decaying reachability with depth. This holds broadly for Erdos–Renyi-like and many sparse random-graph ensembles; structured patterns behave similarly once random long-range contacts exceed a critical rate.
- Depth-limited constraint: A path from a prompt token u to a target v exists through the stacked L layers if and only if the graph distance d(u, v) in the per-layer connectivity (collapsed across positions but respecting direction across layers) is at most L. Thus, two conditions are necessary for robust ICL: (i) percolation (z > 1), and (ii) low typical distances such that P[d(u, v) ≤ L] is high for relevant u, v.

Low-stretch paths
- Chemical distance and “stretch”: Let d(u, v) be the shortest path length from u to v in the one-layer attention graph G(τ) (for stationary patterns) or in a representative layer for nonstationary patterns. A low-stretch route means d(u, v) is close to the metric lower bound imposed by the topology (e.g., O(log n) in small-world regimes). If the architecture imposes local windows of width w with occasional long-range links, the presence of sufficiently many random long-range edges reduces the diameter and d(u, v) from O(n/w) to O(log n), enabling traversal within L for reasonable depths.
- Weighted stretch: Replace thresholded edges with effective costs c_e = −log a_l,h(j ← i). Shortest-path costs approximate the negative log-probability of signal transmission. Low weighted stretch ensures that gradients and information from prompts reach targets without being exponentially attenuated.

Information-theoretic consequence
- Subcritical regime (z ≤ 1): Under mild independence assumptions across layers and positions, the probability that any path of length ≤ L connects a random prompt token u to a target v decays as exp(−κL) for some κ > 0. Consequently, I(target; prompt | model parameters) is upper-bounded by a term that decays exponentially with L in the absence of residual self-copy paths carrying prompt content to v. Training cannot compensate for the absence of such paths.
- Supercritical, low-diameter regime (z > 1 with small d): The expected number of edge-disjoint u→v paths grows exponentially with L up to saturation by n, yielding robust mutual-information transfer and enabling learning algorithms implemented in-context (e.g., retrieval + induction), even under noise or pruning.

Operationalizing z and d in trained models
- Choose τ by calibrating a per-head transmission threshold such that edges represent stable, above-noise flows; alternatives include per-node top-k or mass-based thresholds.
- Compute z_l(τ) empirically from attention maps on task-agnostic inputs; report z(τ) as the geometric mean across layers, and profile its depthwise variation.
- Estimate d(u, v) by shortest-path computations on G(τ) between prompt and target segments; report the fraction of pairs with d(u, v) ≤ L and the distribution of weighted stretches.

Predictions
- P1 (Percolation threshold): ICL metrics (e.g., few-shot accuracy) exhibit a sharp increase as z crosses 1, holding depth fixed.
- P2 (Depth–distance tradeoff): For fixed z > 1, increasing L improves ICL until typical d(u, v) ≤ L; beyond that, gains saturate.
- P3 (Head and window scaling): Increasing head count or window size raises z and reduces d(u, v), shifting the ICL threshold to smaller L.
- P4 (Structured sparsity): Architectures with only local windows (no random long-range edges) require L ≈ Θ(n/w) to achieve ICL; adding a small number of random long-range edges per node collapses this to L ≈ Θ(log n).
- P5 (Ablation robustness): Above threshold, pruning a small fraction of edges that preserves z > 1 does not destroy ICL; pruning that reduces z below 1 collapses ICL abruptly.

Experiments (falsification plan)
Goals
- Test for threshold-like behavior in ICL as a function of per-layer percolation and graph distance.
- Disentangle percolation (existence of macroscopic connectivity) from low-stretch routing constraints.

Models
- Sparse-attention Transformers with controllable connectivity (sequence length n = 1k–8k; depth L = 4–48; heads H = 1–32).
- Attention patterns:
  1) Random Erdos–Renyi edges per head with expected out-degree k; tune k to span subcritical to supercritical regimes.
  2) Local banded windows of width w with s random long-range contacts per token; vary s from 0 to 4.
  3) Block-sparse patterns (sliding windows + block jumps) with a small random rewiring probability p.
- Edge control: Implement via static masks or learned sparse gates (e.g., top-k) constrained to target k, s, p. Ensure residual self-edges remain.

Training
- Train on standard next-token prediction over a mixture of corpora plus synthetic in-context tasks embedded as prompts: linear regression, sparse parity, majority label classification, key–value retrieval and mapping, and small language tasks with few-shot examplars.
- Keep total parameter count and optimizer constant across sparsity conditions; match compute budgets.

Metrics
- Graph metrics: For each trained model and τ, estimate z_l(τ), z(τ), fraction of nodes in the giant component, distance distribution d(u, v) between prompt and target segments, and weighted stretch summaries.
- ICL metrics: Few-shot accuracy vs number of demonstrations, sample efficiency (slope of accuracy vs shots), and calibration of predictions dependent on prompt labels vs shuffled prompts.
- Coupling: Plot ICL metrics against z(τ) and against P[d(u, v) ≤ L], not just against architectural hyperparameters.

Falsification tests
- T1 (No threshold): If ICL improves smoothly with k without a kink near z ≈ 1 or without correspondence to P[d ≤ L], the theory is falsified.
- T2 (Distance independence): If ICL persists in regimes where d(u, v) > L for most pairs, falsify low-stretch necessity.
- T3 (Subcritical success): If strong ICL occurs robustly when z < 1 across layers (for all reasonable τ), falsify percolation sufficiency.
- T4 (Pruning invariance): If heavy pruning that drives z below 1 does not degrade ICL, falsify the threshold claim.

Ablations and controls
- Depth sweep at fixed z to confirm depth–distance tradeoff.
- Head and window sweeps isolating z vs d effects.
- τ sensitivity analysis to ensure conclusions are not artifacts of thresholding.
- Negative controls: tasks solvable without prompt dependence (e.g., unigram baselines) to show that LM loss can improve without ICL.

Discussion
This percolation view reinterprets ICL as a connectivity-limited communication problem over the attention graph. It explains why:
- Adding heads or widening windows produces sharp ICL gains: both raise z and reduce typical distances.
- Sparse local attention without long-range edges struggles at long context lengths unless depth scales with sequence distance.
- ICL successes coincide with the appearance of distinct circuits in mechanistic interpretability: supercritical regimes admit many stable routes for specialized information flow.

The framework integrates with small-world network theory: modest random rewiring drastically reduces distance, enabling effective ICL at moderate depths. It also resonates with observations of emergent abilities: phase-transition-like improvements naturally arise when a structural control parameter crosses a critical threshold.

Limitations
- Modeling assumptions: We idealize attention edges as independent across layers and positions and use thresholded graphs; real attention is input-dependent, correlated across heads, and continuous-valued.
- Directed percolation specifics: Exact thresholds depend on the graph ensemble; z ≈ 1 is schematic. Structured masks can deviate from classic ER behavior.
- Mutual information bounds: Our arguments are heuristic; formal bounds require precise stochastic models of attention and training dynamics.
- τ selection: Results may depend on the thresholding scheme; alternative definitions (e.g., mass-constrained paths) may refine z and d estimates.
- Generality: Conclusions are most direct for tasks requiring routing from prompt to target; tasks solvable via global statistics or self-cues may be less sensitive to percolation.

Conclusion
We advance a structural theory of in-context learning in Transformers: a percolation threshold in the per-layer attention graph governs the emergence of strong ICL, with low-stretch paths enabling effective routing within depth. The theory predicts sharp, falsifiable transitions as connectivity parameters cross criticality and as depth suffices to cover typical graph distances. By reframing ICL as a graph phase transition, this work provides actionable diagnostics, principled sparsity design targets, and a unified explanation for disparate empirical phenomena surrounding emergent ICL in modern Transformers.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
