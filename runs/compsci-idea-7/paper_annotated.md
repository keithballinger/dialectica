Minor Revisions
Justification: The framework is clear and testable, with sensible estimators and predictions. However, some assumptions (local tree-likeness, weighted non-backtracking extension) and estimator robustness (head aggregation dependence, finite-size effects) need sharper caveats and empirical validation guidance.

Paper (Markdown):
# A Directed-Percolation Theory of In-Context Learning: Spectral Thresholds and Depth-Constrained Routing in Transformers
> Note: This title proposes a physics-inspired view of how Transformers learn from prompts (in-context learning) by modeling information flow as “percolation” (like fluid spreading) through attention connections. It claims that two measurable conditions—how paths proliferate (spectral thresholds) and whether paths fit within model depth—predict when ICL emerges.

## Abstract
We propose a structural theory of in-context learning (ICL) in Transformers that treats information propagation as directed percolation on the attention-induced communication graph. The theory posits two necessary conditions for robust ICL over a task distribution: (i) a supercritical path-proliferation condition R > 1, defined via a backtracking-suppressed line-graph operator on the (data-averaged) multi-head attention graph, and (ii) a depth-constrained routing condition requiring that prompt-to-target paths fit within model depth L. We operationalize these criteria with scalable estimators for spectral growth, path-length distributions, and min-cut capacity using Jacobian-based edge capacities. We present falsifiable predictions (including finite-size scaling) and a preregistered plan. The framework unifies observations of abrupt ICL emergence, head-count effects, sparsity, and small-world attention under a single quantitative model.
> Note: Key ideas:
> - Treat the model’s attention as a directed graph (edges point from source token to target token across layers).
> - Two conditions predict ICL: (1) R > 1 means information-carrying paths multiply rather than die out; (2) paths from the prompt to the answer must fit within the number of layers L.
> - “Backtracking-suppressed line-graph” removes trivial “looping on self” effects so the growth rate R reflects genuine propagation. “Spectral” refers to eigenvalues (growth rates).
> - Estimators: spectral radius for growth (R), shortest-path lengths for routing, and max-flow with Jacobian-derived edge capacities for usable information.
> - “Falsifiable predictions” means these can be checked and potentially disproven by experiments.

## 1. Introduction
Transformers often acquire ICL: solving tasks specified in the prompt without parameter updates. We lack predictive criteria that specify when ICL emerges as a function of attention sparsity, head count, and depth.
> Note: In-context learning (ICL) means a model learns from examples in the input prompt at inference time. The gap: we don’t have formulas telling when ICL appears given architecture choices (how sparse attention is, number of heads, depth).

We model attention as an input-dependent, layered, directed, weighted graph. Two structural constraints govern ICL capability over a data distribution:
- Supercritical connectivity (R > 1): long-range proliferation of non-degenerate information paths.
- Depth-constrained routing: prompt-to-target paths fit within L layers.
> Note: Model abstraction:
> - “Layered, directed, weighted graph”: nodes are token positions per layer; edges carry attention weights from one layer’s positions to the next.
> - “Non-degenerate paths” exclude trivial immediate reversals/stacked self-loops. “Supercritical” (R > 1) means each step on average creates more viable path weight than it loses.
> - “Depth-constrained routing”: the minimum number of layers needed for a path must be less than the model’s layers L.

We derive estimators and falsifiable predictions to assess these constraints and guide design of sparse/small-world attention.
> Note: They propose measurable quantities and testable hypotheses to inform architecture choices (e.g., adding long-range edges to create “small-world” graphs with short paths).

## 2. Communication graph and aggregation
Setup. Consider a decoder-only Transformer with sequence length n, depth L, and H heads per layer. For input x, attention from source position i at layer l−1 to target position j at layer l and head h is a_l,h(j|i,x) ∈ [0,1], normalized per target j. Causal masking enforces i ≤ j.
> Note: Definitions:
> - n: number of tokens. L: number of layers. H: number of attention heads per layer. x: an input sequence.
> - a_l,h(j|i,x): attention weight (probability-like) from position i (previous layer l−1) to position j (current layer l) for head h, given input x. Sum over all i for fixed j is 1 (normalization).
> - “Causal masking” (decoder-only) prevents attending to future tokens: i ≤ j.

Edges. Define per-head weighted adjacency W_l,h(x) with entries W_l,h[i→j] = a_l,h(j|i,x). Residual connections are modeled as self-edges i→i.
> Note: W_l,h is a matrix of edge weights per head, where entry (i→j) equals the attention weight. Residual connections are represented as edges that go from a node to itself, allowing skip of transformation.

Aggregating heads. We require an effective per-layer weight W_l. We aggregate per input x using:
- U_l(i→j|x) = 1 − Π_h (1 − W_l,h[i→j]) (soft-OR).
> Note: U_l(i→j|x) combines multiple heads into a single edge weight using a “soft-OR”: it’s high if any head strongly connects i to j. Π_h denotes product over heads h.

Bounds independent of head independence:
- max_h W_l,h[i→j] ≤ U_l(i→j|x) ≤ min(1, Σ_h W_l,h[i→j]).
> Note: Inequalities show that the soft-OR lies between the maximum single-head weight and the (capped) sum of head weights. Σ_h is sum over heads; max_h is the maximum over heads. The cap at 1 ensures probabilities don’t exceed 1.

We carry both the soft-OR value and these bracketing bounds through diagnostics to assess sensitivity to head dependence.
> Note: Because heads may not act independently, they report results using soft-OR and the upper/lower bounds to quantify uncertainty from head interactions.

Data averaging. Define the annealed operator W̄_l = E_x[U_l(·|x)] (aggregate within input, then average). We also analyze quenched quantities that keep input dependence (Section 6).
> Note: W̄_l is the per-layer edge-weight matrix averaged over inputs x (E_x denotes expectation over the data distribution). “Annealed” uses average structure; “quenched” keeps variability per input.

## 3. Backtracking-suppressed spectral growth in causal, layered graphs
Why suppress backtracking? Spectral thresholds for percolation in heterogeneous graphs are governed by growth of non-degenerate walks (walks that do not immediately reverse). In causal, layered DAGs with i ≤ j, immediate reversals do not exist except for stacking self-loops (i→i→i). Suppressing those trivial walks avoids spuriously inflating growth via residual accumulation.
> Note: The growth rate should reflect meaningful propagation, not repeated self-connections (residuals). “Percolation” threshold depends on counts of forward-moving walks. A DAG (directed acyclic graph) with causal masking already prevents backward steps; only repeated self-loops remain to be filtered out.

Backtracking-suppressed line-graph operator. For layer l with effective adjacency W̄_l, let E_l = {(i→j): W̄_l[i→j] > 0}. Define L_l ∈ R^{|E_l|×|E_l|} by
- L_l[(i→j),(j→k)] = W̄_l[j→k] if k ≠ i; 0 otherwise.
With causal masking, k < j is disallowed, so k = i can only occur when i = j = k; thus L_l removes stacked self-loop transitions (i→i)→(i→i). For general directed graphs (e.g., with bidirectional edges), L_l reduces to the weighted Hashimoto non-backtracking operator.
> Note: Definitions:
> - W̄_l: averaged adjacency (edge-weight) matrix at layer l.
> - E_l: set of directed edges with nonzero weight at layer l.
> - L_l: “line-graph” operator whose states are edges; it maps a current edge (i→j) to a next edge (j→k) with weight W̄_l[j→k], except it forbids immediate reversal (k ≠ i).
> - The operator dimensionality is |E_l|×|E_l| (number of edges squared).
> - With causal masking, the only disallowed backtracking is stacked self-loops (i→i) followed by (i→i). Hashimoto operator is the standard non-backtracking matrix; this is its weighted, forward-only version.

Layer aggregation and reproduction factor. Let R_l = ρ(L_l) be the spectral radius. Define the geometric-mean reproduction factor
- R = exp((1/L) Σ_{l=1}^L log R_l),
which matches the top Lyapunov exponent for i.i.d. (or stationary ergodic) layers and stabilizes heterogeneous products.
> Note: Definitions:
> - ρ(L_l): spectral radius (largest absolute eigenvalue) of L_l; it quantifies per-layer growth of non-backtracking walks.
> - R: geometric mean of R_l across layers; exp is the exponential function; Σ is sum; log is natural logarithm; L is number of layers.
> - This equals the top Lyapunov exponent (asymptotic growth rate) for random products when layers are independent and identically distributed (i.i.d.) or stationary/ergodic.

Assumptions. We consider a random layered graph model induced by data averaging with:
- A1 (Sparsity/local tree-likeness across layers): bounded average out-degree, vanishing short non-self cycles in the large-n limit (within layers).
> Note: A1 means each node has only a few outgoing edges on average, and the graph locally looks like a tree (few short loops), especially as sequence length n grows.

- A2 (Bounded weight moments): E[W̄_l[i→j]^2] < ∞.
> Note: A2 requires finite second moments of edge weights, ensuring no extreme heavy tails that would break spectral estimates.

- A3 (Stationarity/ergodicity): {W̄_l} is stationary across layers (or admits an ergodic decomposition).
> Note: A3 assumes that the distribution of layer graphs doesn’t drift with depth (or can be decomposed into stable components), enabling law-of-large-numbers behavior for products across layers.

Proposition 1 (Annealed threshold; formal statement). Under A1–A3, there exists a critical threshold R⋆ = 1 such that:
- If R ≤ 1, the expected total weight of non-degenerate prompt-to-target walks of length L decays at least exponentially with L.
- If R > 1, the expected weight remains bounded away from zero as L increases (and grows for sufficiently large L), yielding a positive probability of long-range communication via non-degenerate paths.
Moreover, for strictly causal layered DAGs the operator L_l equals the non-backtracking operator restricted to forward edges and with self-loop stacking suppressed.
> Note: Interpretation:
> - R⋆ (critical value) equals 1: below it (R ≤ 1), useful path weight dies out quickly as depth grows; above it (R > 1), non-degenerate paths persist with nonzero probability and can support long-range information flow.
> - “Expected total weight” aggregates weights across all non-backtracking paths.
> - The specialized operator matches the standard non-backtracking operator in this causal setting after removing stacked self-loops.

Proof sketch. Map edge-weighted paths to walks on the line graph of the layered DAG; immediate backtracking corresponds to (i→j) followed by (j→i). Under causality, only stacked self-loops remain; removing them yields L_l. Standard results for the growth of non-backtracking walks and percolation thresholds in sparse random graphs (e.g., Hashimoto; Krzakala et al., 2013; Bordenave, Lelarge, and Massoulié, 2015) extend to weighted settings with bounded moments and local tree-likeness by replacing counts with weighted branching factors. Stationarity across layers yields a multiplicative ergodic theorem argument giving the geometric mean criterion R > 1 for supercritical growth.
> Note: Logic:
> - Convert path counting to non-backtracking walk dynamics on the line graph.
> - Use known theorems for sparse graphs to relate spectral radius to percolation thresholds, adapted to weighted edges under A1–A2.
> - Apply multiplicative ergodic theory (Lyapunov exponents) under A3 to justify using the geometric mean R as the threshold.

Remark. In strictly causal settings without residual self-loops, L_l and adjacency have the same spectral threshold; suppressing self-loop stacking avoids trivial inflation from residual accumulation.
> Note: If there are no residual self-loops, you don’t need suppression; the threshold from adjacency and from the non-backtracking operator coincide. Residual self-loops can otherwise inflate growth estimates.

## 4. Depth-constrained routing
Define d(u,v) as the minimal number of layers needed to route from prompt token u to target v on the layered graph defined by {W̄_l}, with edge cost 1 (or more generally cost proportional to −log W̄_l). Define a characteristic path length d_char (e.g., median over prompt-target pairs).
> Note: Definitions:
> - d(u,v): shortest number of layers (hops) from source token u to destination token v, treating each inter-layer edge as cost 1. A weighted variant uses cost −log W̄_l (penalizing weak edges).
> - d_char: a typical distance (e.g., median) over all prompt-target pairs.

Depth-constrained routing condition:
- d_char < L.
> Note: Condition says typical source-to-target paths must fit within the model’s depth L for ICL to succeed.

Local attention implies d_char = Ω(n). Adding sparse long-range edges per layer yields small-world behavior with d_char = O(log n) under standard conditions (Watts–Strogatz-type augmentations), allowing ICL at significantly smaller L.
> Note: With only local windows (no long-range links), path lengths scale linearly with sequence length n (Ω(n) means grows at least proportionally). Adding a few random long-range edges per layer creates “small-world” graphs where distances scale like log n, so fewer layers are needed.

## 5. Information capacity via Jacobian-based edge capacities
Reachability (R > 1) is necessary but not sufficient. We define edge capacities using pathwise sensitivity to quantify usable information flow.
> Note: Even if paths exist and proliferate (R > 1), the model might not transmit useful signal due to interference or value mixing. “Capacity” estimates how much change at a source can affect a target (sensitivity).

Edge capacities from Jacobians. Let h_{l,j} be the post-attention hidden state at layer l, position j. Define
- c_l(i→j) = E_x[|∂h_{l,j}(x)/∂h_{l-1,i}(x)|],
estimated via directional derivatives and Hutchinson’s trace estimators. This captures the effective channel strength, incorporating attention and value mixing. Alternatives include squared sensitivities or task-conditioned Jacobians (e.g., ∂ℓ/∂h).
> Note: Definitions:
> - h_{l,j}: vector of activations after attention at layer l, token position j.
> - ∂h_{l,j}/∂h_{l-1,i}: Jacobian (matrix of partial derivatives) measuring how a small change at (l−1,i) affects (l,j). The absolute value |·| here denotes an entrywise norm or magnitude; the paper treats it as a scalar sensitivity (specify norm in practice).
> - c_l(i→j): expected sensitivity over inputs x; larger means a stronger information channel.
> - Estimation uses Jacobian-vector products (directional derivatives) and Hutchinson’s estimator (a randomized method to estimate trace- or norm-like quantities) with low overhead.
> - Variants: use squared derivatives or condition on a task loss ℓ.

Layered min-cut bound. Form a layered DAG with capacities c_l(i→j). For source set S (prompt tokens) and sink set T (targets), the maximum s–t flow upper-bounds the total Jacobian-mediated information transfer from S to T (by submultiplicativity/data processing). We predict:
- In the supercritical regime, s–t min-cut scales up with L until saturation; in subcritical regimes, it vanishes with L.
> Note: Build a flow network where each edge capacity equals sensitivity. The min-cut (smallest total capacity you must cut to disconnect S from T) bounds how much information can pass. Prediction: above threshold, capacity grows with depth; below, it shrinks.

Empirical practicality. Compute c_l via mini-batch estimates with low-overhead Jacobian-vector products; run max-flow (Dinic or push-relabel) per model on sparsified graphs.
> Note: Practical recipe: estimate sensitivities with standard automatic differentiation tricks; sparsify the graph (keep strongest edges) and compute max-flow using efficient algorithms (Dinic, push–relabel).

## 6. Annealed versus quenched thresholds
Define per-input operators U_l(x), L_l(x). Two thresholds:
- Annealed: build W̄_l = E_x[U_l(x)], then L̄_l = L_l(W̄_l), and R_annealed from {L̄_l}.
> Note: Annealed procedure: average edges over inputs first (U_l → W̄_l), then form non-backtracking operator L̄_l and compute R_annealed via spectral radius aggregation.

- Quenched: top Lyapunov exponent of random products:
  R_quenched = exp(lim_{L→∞} (1/L) E_x[log ||L_L(x_L)···L_1(x_1)||]).
> Note: Definitions:
> - R_quenched measures growth when multiplying per-input operators in sequence.
> - ||·|| denotes a matrix norm (e.g., operator norm); log is natural log; E_x averages over input sequences; lim_{L→∞} takes large-depth limit; exp converts back from logs.

Typically R_annealed ≥ R_quenched; increased input variance near criticality widens the annealed–quenched gap. We estimate both, report their difference, and analyze the distribution of per-layer growth factors across inputs.
> Note: Averaging first (annealed) usually overestimates growth compared to averaging logs (quenched). Near the threshold, variability makes the gap larger. Reporting both quantifies uncertainty.

## 7. Falsifiable predictions
- P1 (Spectral threshold): As a control parameter (e.g., sparsity, head count, attention temperature) varies, ICL accuracy exhibits a change-point at R ≈ 1. A logistic model for accuracy vs. R yields a threshold parameter near 1 with tight CIs; finite-size scaling collapses performance when plotted against (R − 1)L^β for some β > 0.
> Note: Expect a sharp transition in performance when R crosses 1. “Logistic model” fits an S-shaped curve; “CI” is confidence interval. “Finite-size scaling” means curves for different depths align when rescaled by (R−1)L^β, indicating universal behavior.

- P2 (Depth-distance tradeoff): For supercritical models (R > 1), increasing L improves ICL until L ≈ d_char, after which gains saturate (plateau detected by a slope change indistinguishable from zero within CI).
> Note: Deeper models help until depth matches typical path length; beyond that, no further gains—detectable by near-zero slope within statistical uncertainty.

- P3 (Small-world benefit): Adding k random long-range edges per layer (fixed FLOPs) reduces d_char ≈ O(log n/k), shifting the ICL threshold to smaller L and larger accuracy at fixed L. The shift magnitude scales with measured change in d_char.
> Note: Adding a few random connections per layer dramatically shortens paths (d_char ∝ log n divided by k), improving ICL at the same depth or enabling shallower models.

- P4 (Pruning phase change): Random edge pruning that keeps R > 1 degrades ICL smoothly; dropping R below 1 produces a sharp collapse identified by change-point detection.
> Note: You can prune edges without killing ICL if R stays >1; crossing below 1 triggers a sudden performance drop (detectable statistically).

- P5 (Temperature control): Modulating attention temperature shifts R across 1 and induces a reproducible ICL transition without retraining.
> Note: Attention temperature (scaling the softmax) controls sparsity/sharpness; tuning it at inference can push R over/under 1 and toggle ICL.

- P6 (Heads vs. overlap): Gains from more heads track increases in R (diminishing returns with overlapping edges); bracketing aggregators (max vs. capped-sum) bound the effect size.
> Note: More heads can increase connectivity and R, but if they focus on the same edges the benefit saturates. Using different head-aggregation bounds provides upper/lower performance estimates.

- P7 (Annealed–quenched gap): Near criticality, annealed thresholds anticipate emergence relative to quenched; input-to-input variance of per-layer growth spikes near the transition.
> Note: The annealed metric predicts ICL earlier than quenched. Also, variability across inputs is highest near R ≈ 1, a hallmark of criticality.

## 8. Estimation and computational costs
- Participation ratio (PR): For each head h, PR_l,h(i) = 1 / Σ_j a_l,h(j|i)^2; relate distributional summaries to R.
> Note: Definitions:
> - PR measures how many targets j a source i effectively attends to (higher PR = more spread). Σ_j sums over targets.
> - Aggregated PR statistics can correlate with connectivity growth R.

- Spectral growth:
  - Annealed: build W̄_l and L̄_l; estimate R_l via power iteration. Cost O(k|E_l|) for k iterations with sparse edge-level multiplication.
> Note: Compute dominant eigenvalue via repeated multiplications (power iteration). |E_l| is number of edges at layer l; O(k|E_l|) indicates linear cost in edges times iteration count.

  - Quenched: sample sequences {L_l(x)} and estimate the Lyapunov exponent via products with periodic QR reorthogonalization; cost O(kL|E|) per sequence; reduce variance via bootstrapping and subgraph sampling.
> Note: Multiply random layer operators and stabilize numerics with QR decomposition. |E| is typical edge count per layer; k controls iterations or repeats; bootstrapping/subsampling stabilizes estimates.

  - Scalability: use sparsification, graph coarsening, or Nyström/sketching for approximate spectra with error bounds.
> Note: Reduce computational load by approximating the graph/eigenvalues while tracking approximation error.

- Shortest paths: Compute d(u,v) on the layered DAG using multi-source BFS/Dijkstra. Cost O(|E| + |V|) (unweighted) or O(|E| log |V|) (weighted).
> Note: |V| is number of nodes (token positions across layers). BFS works for unit costs; Dijkstra for weighted edges.

- Capacities and min-cut: Estimate c_l(i→j) via Jacobian-vector products; run max-flow using Dinic (O(|E|√|V|)) or push-relabel; exploit layer structure for faster blocking flows.
> Note: Standard max-flow algorithms scale near-linearly in edges for practical graphs; layered structure can accelerate computations.

- Robustness to head aggregation: Repeat all diagnostics with soft-OR, max-over-heads, and capped-sum to bound uncertainty from head dependence.
> Note: Reporting multiple head-aggregation schemes provides sensitivity analysis against head correlation assumptions.

## 9. Preregistered experimental design
Architectures.
- Random sparse attention (Erdős–Rényi masks) sweeping sparsity.
> Note: Use random graphs controlling the probability of edges to vary sparsity and test thresholds.

- Small-world attention: local windows + k random long-range edges per layer.
> Note: Combine nearby attention with a few random long-range links to create small-world routing.

- Head count sweeps with fixed per-head dimension; inference-time temperature sweeps.
> Note: Vary number of heads while keeping head size constant; adjust attention temperature at inference to move R.

Controls.
- Fix total parameters, training FLOPs, sequence length, datasets, and optimization hyperparameters across conditions.
> Note: Ensure fair comparisons by holding constant model size, training cost, data, and training settings.

- Standardize LayerNorm/residual scales; log activations to detect confounds.
> Note: Keep normalization and residual magnitudes consistent; record activations to check for scale-related artifacts.

Tasks.
- ICL probes: key–value retrieval, copy/induction, algorithmic tasks (addition, parentheses), and few-shot classification across heterogeneous distributions.
> Note: Test a range of tasks that exercise ICL mechanisms (retrieval, copying patterns, simple algorithms, and classification with few examples).

Analyses.
- Estimate R_annealed, R_quenched, d_char, min-cut per model; report CIs.
> Note: For each model, compute the key metrics with confidence intervals to quantify uncertainty.

- Change-point detection (e.g., PELT) for ICL onset; finite-size scaling across {n, L}.
> Note: Use statistical methods (like PELT) to detect sharp transitions; test scaling laws by varying sequence length n and depth L.

- Variance of per-layer growth across inputs; controlled ablations to test P3–P6.
> Note: Measure instability near criticality and run targeted interventions (small-world edges, pruning, temperature, head count) to confirm predictions.

## 10. Related work
- Non-backtracking spectra and percolation thresholds: Hashimoto; Krzakala et al., 2013; Bordenave, Lelarge, Massoulié, 2015; Newman, 2018.
> Note: Prior theory links non-backtracking eigenvalues to percolation/giant component formation, underpinning the R > 1 threshold.

- Small-world routing and diameter: Watts & Strogatz, 1998; per-layer small-world constructions.
> Note: Classic results show adding a few long-range links dramatically reduces graph distances, motivating Section 4.

- Sparse Transformers and efficiency: Child et al., 2019; routing in sparse attention.
> Note: Practical architectures with sparse attention align with the proposed graph-based view.

- Attention flow and Jacobians: Abnar & Zuidema, 2020; gradient-based attribution and path sensitivity; dynamical isometry and signal propagation (Pennington et al., 2017).
> Note: Related work uses gradients/Jacobians to trace information flow and analyze signal propagation stability.

- Mechanistic ICL: induction circuits and attention patterns (e.g., Olsson et al., 2022).
> Note: Empirical studies identify specific attention circuits enabling ICL, which this model abstracts at the graph level.

## 11. Limitations
- Applicability of spectral thresholds depends on sparsity and local tree-likeness; dense attention may require alternative random-matrix analyses.
> Note: The main theory may not hold when attention is dense with many short loops; different mathematics (dense random matrices) may be needed.

- R > 1 is necessary but not sufficient; interference and value transformations can block computation despite reachability. Jacobian capacities mitigate but do not eliminate this gap.
> Note: Even if paths proliferate, the content can be corrupted; capacity estimates help but don’t fully capture computation quality.

- Head independence is an approximation; bracketing aggregators bound but do not remove dependence.
> Note: The aggregation assumes limited head dependence; using bounds (max vs. capped-sum) provides uncertainty ranges but not exactness.

- Estimation error for large models requires careful numerical control; finite-size and training nonstationarity can blur transitions.
> Note: Practical computation introduces noise; model size and evolving training distributions can smear sharp thresholds.

## 12. Conclusion
ICL emergence can be predicted by two structural conditions: supercritical backtracking-suppressed spectral growth (R > 1) and depth-constrained routing (d_char < L). Coupled with Jacobian-based capacity diagnostics, these criteria yield sharp, testable predictions about when ICL appears and how it depends on architecture. This framework aims to shift emergent-ability studies from descriptive to predictive, informing principled design of sparse and small-world attention.
> Note: Summary: When non-backtracking path growth exceeds 1 and paths fit within depth, ICL should appear. Measuring both, plus sensitivity-based capacity, can predict and guide architecture choices (e.g., sparsity, small-world links, head count) with testable outcomes.
