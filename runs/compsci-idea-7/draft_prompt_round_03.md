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
        Minor Revisions

The paper presents a novel and compelling theory connecting in-context learning to percolation phenomena in the attention graph. The core thesis is strong and supported by a rigorous, falsifiable experimental plan. Revisions are needed to improve precision and clarity. The relationship between the percolation threshold (z>1) and the low-stretch path requirement needs to be more carefully articulated; one does not guarantee the other. The definition of distance is ambiguous, conflating single-layer graph distance with the multi-layer path length required for information to propagate within the model's depth. Key terminology, such as "reproduction number" and "chemical distance," should be replaced with more standard terms from computer science and graph theory (e.g., "branching factor," "shortest-path distance") to improve accessibility.

### Revised Draft

# A Percolation Threshold Governs In-Context Learning in Transformers

## Abstract

We propose a structural theory of in-context learning (ICL) in Transformers. We model the attention mechanism as inducing a directed graph at each layer, where information propagates from prompt tokens to target tokens along paths traversing these graphs. We identify two necessary conditions for strong ICL to emerge. First, the average branching factor *z*—the expected number of tokens a single token attends to—must exceed a directed percolation threshold of *z* = 1. This ensures the existence of macroscopic, depth-spanning communication pathways. When *z* ≤ 1, paths from prompts to targets are exponentially rare with depth, bounding the mutual information that predictions can extract from prompts. Second, the architecture must support low-stretch routing, meaning the shortest path length from a typical prompt token to a target token, measured across layers, must not exceed the model's depth *L*. When both conditions are met (*z* > 1 and short paths exist), the network admits exponentially many edge-disjoint routes, enabling robust ICL. This theory yields falsifiable predictions: threshold-like ICL gains as *z* crosses 1; depth–connectivity tradeoffs governed by the graph's diameter; and a collapse of ICL when path lengths are forced to exceed depth. We outline an experimental program using sparse-attention Transformers with controlled connectivity to test for this phase-transition-like behavior.

## Introduction

Transformer models exhibit in-context learning (ICL), the ability to perform new tasks specified in a prompt without parameter updates. Despite its importance, the structural conditions enabling ICL remain unclear. Prior accounts focus on meta-learning dynamics or specific induction circuits but do not predict when ICL should suddenly emerge or collapse as a function of architectural parameters like sparsity, head count, or depth.

We propose a graph-theoretic account: attention induces a directed, per-layer communication graph over the sequence. For ICL to occur, information must propagate from prompt tokens to the prediction site along a path of attention edges across the model's layers. This requires two conditions to be met.

Our core claims are:
1.  **Percolation Enables Connectivity**: Define *z* as the average per-layer branching factor (expected out-degree) of the attention graph. A directed percolation transition at *z* = 1 governs the existence of macroscopic, depth-spanning paths from prompts to targets. Below this threshold, prompt information cannot reliably reach the prediction site.
2.  **Low Stretch Enables Routing**: The shortest path from a prompt token *u* to a target token *v* must be achievable within the model's depth *L*. Sparse patterns that keep *z* > 1 but create large graph diameters (e.g., purely local attention) inflate path lengths beyond *L*, suppressing ICL.
3.  **Falsifiable Thresholds**: Families of sparse-attention Transformers should exhibit phase-transition-like ICL gains precisely when *z* crosses 1, provided the low-stretch condition is met.

This view unifies diverse observations—emergent abilities with scale, sharp gains from added heads, and sensitivity to sparsity masks—under a common structural framework.

## Method

### The Per-Layer Attention Graph

Consider a sequence of *n* tokens processed by a Transformer with *L* layers and *H* heads per layer. For layer *l* and head *h*, let `a_l,h(j ← i)` be the attention weight from source token *i* (at layer *l*−1) to target token *j* (at layer *l*).

We model this as a per-layer directed graph *G_l(τ)*. For a fixed transmission threshold *τ* ∈ (0, 1), we place a directed edge *i* → *j* in *G_l(τ)* if any head *h* has `a_l,h(j ← i) ≥ τ`. The full edge set is the union across heads. Residual connections provide a self-edge *i* → *i* at each layer.

### Two Conditions for In-Context Learning

#### 1. Macroscopic Connectivity (Percolation)

Let *z_l(τ)* be the expected out-degree of a node in *G_l(τ)*. We define the average branching factor *z(τ)* as the geometric mean of *z_l(τ)* across layers. This value, *z*, controls a directed percolation transition.

-   **Subcritical (z ≤ 1)**: In directed random graphs, if the mean out-degree is at most 1, the probability of a path existing between two random nodes decays exponentially with the path length. Information from a prompt token is unlikely to reach a distant target token across many layers.
-   **Supercritical (z > 1)**: When the mean out-degree exceeds 1, a giant component emerges, containing a macroscopic fraction of nodes. This ensures the existence of long-range paths necessary for communication across the sequence and through depth.

#### 2. Depth-Limited Routing (Low-Stretch Paths)

The existence of a path is not sufficient; it must be traversable within the model's *L* layers. A path from a prompt token *u* to a target *v* is a sequence of tokens `t_0, t_1, ..., t_k` where *u* = *t_0*, *v* = *t_k*, and there is an edge `t_{i-1}` → `t_i` in graph *G_i(τ)* for *i* = 1...*k*. The length of this multi-layer path is *k*.

Let *d(u,v)* be the length of the shortest such path. Effective ICL requires that for typical prompt-target pairs (*u,v*), their shortest-path distance satisfies **d(u,v) ≤ L**.

This condition depends on the topology of the per-layer graphs. Architectures with small-world properties (e.g., a mix of local and random long-range attention) have small diameters, yielding short paths (e.g., *d(u,v)* = O(log *n*)). In contrast, architectures with only local windows of width *w* have large diameters, forcing path lengths of *d(u,v)* = O(*n/w*), which quickly exceeds typical model depth *L*.

### Information-Theoretic Rationale

-   **Subcritical (z ≤ 1)**: The probability that any path of length ≤ *L* connects a random prompt token to a target decays as exp(−κL). Consequently, the mutual information *I*(target; prompt | parameters) is exponentially bounded in *L*, precluding robust ICL.
-   **Supercritical, Low-Stretch (z > 1, d(u,v) ≤ L)**: The expected number of edge-disjoint paths from *u* to *v* can grow exponentially with *L*, enabling robust information transfer and the implementation of in-context algorithms (e.g., retrieval + induction).

### Operationalizing Metrics

-   Choose *τ* by calibrating a per-head threshold such that edges represent stable, above-noise information flows.
-   Compute *z_l(τ)* empirically from attention maps and report the geometric mean *z(τ)*.
-   Estimate *d(u,v)* via breadth-first search on the multi-layer graph between prompt and target segments. Report the fraction of pairs with *d(u,v) ≤ L*.

## Predictions

-   **P1 (Percolation Threshold)**: ICL metrics will show a sharp increase as *z* crosses 1, provided the architecture supports paths with *d(u,v) ≤ L*.
-   **P2 (Depth–Distance Tradeoff)**: For a fixed *z* > 1, increasing *L* improves ICL until *L* exceeds the typical shortest-path distance *d(u,v)*, at which point gains saturate.
-   **P3 (Architectural Scaling)**: Increasing head count or adding non-local attention increases *z* and reduces *d(u,v)*, enabling ICL at smaller depths.
-   **P4 (Structured Sparsity)**: Architectures with only local attention require depth *L* ≈ Θ(*n*/*w*) for ICL. Adding a small fraction of random long-range edges per token reduces typical path lengths to Θ(log *n*), enabling ICL at much shallower depths.
-   **P5 (Ablation Robustness)**: For a model in the supercritical regime, pruning edges that keeps *z* > 1 will cause gradual degradation, whereas pruning that pushes *z* below 1 will cause a sudden collapse in ICL performance.

## Experiments (Falsification Plan)

The experimental goal is to test for threshold-like behavior in ICL as a function of our two proposed conditions.

-   **Models**: Use sparse-attention Transformers with controllable connectivity patterns:
    1.  **Random (Erdos–Renyi)**: Tune expected out-degree to span the *z* = 1 threshold.
    2.  **Small-World**: Use local windows plus a variable number of random long-range connections to control graph diameter and thus *d(u,v)*.
-   **Training**: Train models on a standard language modeling objective mixed with synthetic in-context tasks (e.g., retrieval, classification, regression).
-   **Metrics**:
    -   **Graph Metrics**: Empirically measure *z(τ)* and the distribution of *d(u,v)*.
    -   **ICL Metrics**: Evaluate few-shot accuracy on held-out tasks.
    -   **Coupling**: Plot ICL accuracy directly against *z(τ)* and P[*d(u,v) ≤ L*].
-   **Falsification Tests**:
    -   **T1 (No Threshold)**: If ICL improves smoothly with connectivity, without a sharp change near *z* ≈ 1, the percolation theory is challenged.
    -   **T2 (Distance Independence)**: If strong ICL persists in models where typical *d(u,v)* > *L*, the low-stretch necessity claim is falsified.
    -   **T3 (Subcritical Success)**: If strong ICL occurs robustly when *z* < 1 (for all reasonable *τ*), the theory is falsified.

## Discussion

This percolation framework reinterprets ICL as a connectivity-limited communication problem. It explains why adding heads or randomizing sparse attention can produce sharp, emergent gains in ICL: both actions can push the attention graph into the supercritical, low-diameter regime. The theory integrates with small-world network principles, explaining why a few long-range attention edges are so effective. It also provides a structural mechanism for the "phase transitions" observed in emergent abilities of large models.

## Limitations

Our model idealizes attention using a binary threshold *τ*; real attention is continuous and input-dependent. The exact percolation threshold *z* = 1 holds for certain random graph ensembles and is an approximation for structured attention patterns. Our information-theoretic arguments are heuristic. The conclusions apply most directly to ICL tasks requiring explicit data routing from prompt to target.

## Conclusion

We propose a structural theory of in-context learning based on two conditions: the per-layer attention graph must be supercritical (*z* > 1) to ensure connectivity, and it must support low-stretch paths traversable within the model's depth (*d(u,v) ≤ L*) to enable routing. This theory offers a unified, falsifiable explanation for the emergence of ICL and provides principled guidelines for designing efficient, sparse Transformer architectures.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
