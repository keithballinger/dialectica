Minor Revisions

Brief critique
- The core theoretical claims rest on a data-averaged communication graph, `Ê[Uℓ]`, which abstracts away the input-dependent routing critical to ICL. The manuscript should clarify that this model predicts the emergence of the ICL *capability* over a data distribution, not its execution on a specific instance. The variance of graph properties across inputs is a key unaddressed factor.
- The role of value transformations and MLPs is acknowledged as a limitation but downplayed in the core percolation model. The theory is primarily one of token *reachability*, but ICL is about computation. The link between reachability (R > 1) and sufficient information capacity for computation needs to be made more explicit, perhaps by framing the percolation threshold as a necessary but not sufficient condition, with capacity metrics providing a fuller picture.
- The justification for the non-backtracking operator is implicit. Briefly explain its role in modeling path proliferation for a branching-process approximation on a general directed graph, making it a more robust choice than a simple adjacency matrix.
- The "union-of-heads" aggregator `Uℓ = 1 − Πh(1 − Aℓ,h)` assumes independent signal transmission probabilities across heads. This is a strong assumption that should be stated and briefly justified as a tractable first-order model of redundant pathways.
- The term "low-stretch routing" is slightly imprecise. "Depth-constrained routing" or a similar term would be clearer, as "stretch" typically compares graph distance to an external metric (e.g., Euclidean distance), whereas here the constraint is the model's intrinsic depth `L`.

Revised Draft
# A Directed-Percolation Theory of In-Context Learning: Spectral Thresholds and Depth-Constrained Routing in Transformers

## Abstract
We propose a structural theory of in-context learning (ICL) in Transformers, framing it as directed percolation in the attention-induced communication graph. Our theory posits two quantitative, falsifiable conditions for the emergence of robust ICL capability: (i) a supercritical connectivity condition, R > 1, on the spectral radius of the non-backtracking operator associated with the data-averaged, multi-head attention graph; and (ii) a depth-constrained routing condition requiring that prompt-to-target information paths fit within model depth L. Unlike heuristics based on thresholded degree, our spectral criterion is threshold-free and principled for weighted, directed graphs with heterogeneous connectivity. We propose operational metrics (participation-ratio effective degree, spectral radius estimators, path-length distributions, min-cut capacity bounds) and a preregistered experimental plan to test the theory's predictions, including the existence of a sharp phase transition for ICL performance. The theory unifies observations of abrupt ICL emergence and the effects of head count and sparsity into a single structural framework.

## 1. Introduction
Transformers can solve new tasks specified in-context without parameter updates, a capability known as in-context learning (ICL). Despite its importance, we lack a general theory predicting when ICL emerges or fails as a function of core architectural parameters like depth, sparsity, and head count.

We model attention as an input-dependent, layered, directed graph and hypothesize that the ICL *capability* is governed by two structural constraints on this graph, averaged over a data distribution:
1.  **Supercritical Connectivity:** The communication graph must be in a supercritical regime to support robust, long-range information flow from prompt to target.
2.  **Depth-Constrained Routing:** The characteristic path length from prompt tokens to target tokens must be less than the model's depth, `L`.

Our contribution is a falsifiable, quantitative theory that moves beyond task-specific circuit analyses or meta-learning analogies. It provides measurable graph-theoretic diagnostics to predict the onset of ICL, explain its sensitivity to architectural choices, and guide the design of efficient sparse models.

**Contributions:**
- A percolation criterion (R > 1) for ICL capability based on the spectral radius of the non-backtracking operator of the data-averaged attention graph.
- A depth-constrained routing condition connecting characteristic prompt-to-target path lengths to model depth `L`.
- Operational estimators for the theory's core quantities, including participation-ratio effective degree, spectral radius, shortest-path distributions, and information-capacity proxies.
- A preregistered experimental design to test for finite-size scaling and sharp ICL thresholds, controlling for confounders like parameter count and residual/MLP pathways.

## 2. Model: Attention as a Layered, Directed, Weighted Graph
For a sequence of `n` tokens processed by an `L`-layer, `H`-head Transformer, we denote the attention from source `i` (at `ℓ-1`) to target `j` (at `ℓ`) for head `h` and input `x` as `aℓ,h(j | i, x)`. This defines a weighted adjacency matrix `Aℓ,h(x)`. To model the combined effect of multiple heads, we define a per-layer union-of-heads operator:
`Uℓ(i → j; x) = 1 − Πh (1 − Aℓ,h(i → j; x))`
This aggregator models heads as providing redundant pathways, assuming their probabilistic contributions are independent as a first-order approximation.

Our theory concerns the emergence of ICL as a general capability. We therefore analyze the **data-averaged operator** `Ê[Uℓ]`, where the expectation is over a representative data distribution. This removes input-specific details to reveal the average structural properties that enable ICL. Residual connections are modeled as self-edges, while MLP and value-vector transformations are treated as content-modifying operations on this communication backbone, whose effects are assessed via capacity metrics and ablations (§4, §7).

To quantify connectivity without arbitrary thresholds, we use the **participation ratio (PR)**, a measure of the effective number of non-zero elements in a distribution:
- `PRℓ,h(i; x) = 1 / Σj aℓ,h(j | i, x)²`
The data-averaged PR provides a robust measure of effective out-degree.

## 3. Supercritical Connectivity: A Spectral Percolation Threshold
The onset of global connectivity in complex directed graphs is described by percolation theory. For a directed, layered graph, the condition for long-range path existence is governed by the proliferation of paths in a branching-process approximation. This is controlled by the spectral radius of the **non-backtracking operator**, `Bℓ`, associated with the graph `Ê[Uℓ]`. The non-backtracking operator correctly accounts for path dynamics on graphs with cycles and heterogeneous degrees.

We define the layer-wise reproduction number as `Rℓ = ρ(Bℓ)` and the geometric mean across layers as `R = (Πℓ Rℓ)^(1/L)`.

**Proposition 1 (Percolation Threshold for ICL Capability, Informal).**
*Under a locally tree-like approximation of the data-averaged communication graph, the probability of a path existing from a random prompt token to the final target token remains bounded above zero as `L → ∞` if and only if `R > 1`. For `R ≤ 1`, this probability decays exponentially with `L`.*

This establishes `R=1` as a critical threshold. A system with `R > 1` is **supercritical**, possessing the structural backbone for long-range communication. A system with `R ≤ 1` is **subcritical**, where information flow is confined locally.

## 4. Depth-Constrained Routing and Information Capacity
Reachability (`R > 1`) is necessary but not sufficient. Information must also arrive within the model's finite depth and with sufficient fidelity.

### 4.1. Depth-Constrained Routing
Let `d(u, v)` be the shortest path length from a prompt token `u` to a target token `v` in the multi-layer graph defined by `(Ê[Uℓ])ℓ=1..L`. For effective ICL, the characteristic path length `d_char` must satisfy `d_char < L`. Architectures with small-world properties (e.g., local plus random attention) exhibit `d(u,v) = O(log n)`, while purely local attention yields `d(u,v) = Ω(n)`.

### 4.2. Information Capacity
The percolation model ensures path existence but is agnostic to the information transmitted. We complement the structural analysis with a capacity proxy to account for the role of value/MLP transformations and interference. The min-cut max-flow theorem provides a bound on communication bandwidth. We define a per-edge capacity proxy `cℓ(i → j)` derived from attention entropy or value-vector statistics. The minimum prompt-to-target cut capacity serves as a metric for ICL robustness. In the supercritical regime (`R > 1`), we expect this capacity to grow with `L`, whereas it should vanish in the subcritical regime.

## 5. Falsifiable Predictions
- **P1 (Spectral Threshold):** ICL performance, averaged over a task distribution, will exhibit a sharp increase near `R ≈ 1` when varying an architectural parameter (e.g., sparsity). We predict finite-size scaling collapse when plotting performance against `(R − 1)L^β` for some scaling exponent `β`.
- **P2 (Depth-Distance Tradeoff):** For supercritical models (`R > 1`), ICL performance will improve with depth `L` until `L` exceeds the characteristic prompt-target path length `d_char`, after which gains will saturate.
- **P3 (Small-World Advantage):** Adding a few random long-range attention edges to a local attention model will drastically reduce `d_char`, enabling ICL at much smaller `L` for fixed parameter counts.
- **P4 (Graceful vs. Abrupt Degradation):** In a supercritical model, random edge pruning that keeps `R > 1` will cause gradual performance decay. Pruning that pushes `R` below 1 will cause an abrupt collapse.
- **P5 (Temperature Modulation):** Increasing attention temperature lowers PR and `R`. We predict ICL will degrade sharply as temperature is tuned to drive `R` through the critical point at 1.
- **P6 (Head Count Scaling):** ICL performance gains from adding heads will track the resulting increase in `R` (which is sublinear due to head overlap), not the head count itself.

## 6. Operationalization and Estimators
- **Participation Ratio (PR):** Estimate `PRℓ,h(i)` over a validation batch to compute distributions of effective out-degree.
- **Spectral Radius (R):** Construct `Ê[Uℓ]` by averaging attention maps. Estimate `R = (Πℓ ρ(Bℓ))^(1/L)` via power iteration on the non-backtracking operator `Bℓ` for each layer.
- **Shortest Paths:** Construct the layered graph from `(Ê[Uℓ])`; compute the distribution of `d(u, v)` from prompt to target tokens using Breadth-First Search (or Dijkstra's for weighted paths).
- **Capacity Proxy:** Estimate min-cut on the layered graph with edge capacities derived from attention entropy or PR.

## 7. Preregistered Experimental Design
We will test these predictions using controlled experiments with sparse attention Transformers.
- **Architectures:** (1) Random sparse attention (Erdős–Rényi masks) where we sweep sparsity to modulate `R`. (2) Small-world attention (local windows + `k` random edges) where we sweep `k` to modulate `d_char`.
- **Controls:** All comparisons will be made at fixed parameter count, training FLOPs, and sequence length. We will use temperature sweeps as an orthogonal method to modulate `R`. Ablations scaling residual/MLP pathways during evaluation will isolate the attention graph's contribution.
- **Tasks:** A suite of held-out ICL probes (e.g., key-value retrieval, algorithmic tasks, few-shot classification) will assess performance.
- **Analyses:** We will use change-point detection to identify the ICL threshold, perform finite-size scaling analyses by varying `L` and `n`, and analyze the variance of `R(x)` across inputs near the phase transition.

## 8. Related Work
This work synthesizes ideas from several domains. It provides a graph-theoretic foundation for mechanistic ICL theories (e.g., induction heads) and meta-learning perspectives. It connects directly to network science literature on percolation in directed networks (using spectral/non-backtracking criteria) and information flow in deep learning (e.g., dynamical isometry). Unlike prior work on sparse Transformers, which focuses on efficiency, we provide a predictive theory for the emergence of functional capabilities.

## 9. Limitations and Scope
- Our model of the **data-averaged** graph `Ê[Uℓ]` predicts the emergence of the ICL *capability*, not its execution on a specific input `x`. Strong input-dependence may lead to significant variance in `R(x)`.
- The core percolation theory models **reachability**, not computation. The influence of content transformations via value-vectors and MLPs is addressed indirectly via capacity metrics and ablations, not integrated into the spectral threshold itself.
- The branching-process approximation assumes local tree-likeness, while real attention graphs have motifs and correlations that can shift the precise location of the critical threshold.

## 10. Conclusion
We propose that in-context learning is underpinned by principles of directed percolation. The emergence of ICL capability requires both supercritical connectivity (`R > 1`) and depth-constrained routing (`d_char < L`) in the attention communication graph. This theory offers a unified, structural explanation for the abrupt emergence of ICL and its dependence on architecture. By providing quantitative metrics and a clear experimental program, we aim to transform the study of emergent abilities from an observational to a predictive science.
