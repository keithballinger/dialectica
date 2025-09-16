You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Draft:
        Minor Revisions

The paper presents a novel and plausible method, CP-KVP, for reducing LLM prefill latency by transferring KV cache states between semantically similar prompts. The core hypothesis—that the functional effect of a cache can be linearly projected between related prompts—is compelling. The proposed experimental protocol is rigorous, with clear falsification criteria, and the use of simple linear models makes the approach practical and reproducible.

The required revisions are minor and focus on improving notational clarity and precision in the Method section. Specifically, the loss functions for the compression and projection maps should be expressed with more formal mathematical notation to avoid ambiguity. Additionally, minor rephrasing throughout would strengthen the paper's framing and impact.

---
### Revised Draft
Title
Warm-Starting Transformer Inference with Linear KV Cache Projection

Abstract
The prefill stage for autoregressive language models is a primary driver of end-to-end latency. While KV cache reuse can eliminate this cost, it is only effective when a new prompt shares an exact prefix with a cached one. We propose a method to generalize cache reuse to semantically similar prompts. Our approach learns a linear projection that maps the KV cache of a previously seen source prompt to an initial KV cache for a new target prompt. This "warm-start" allows the model to generate the first token without a full prefill pass, significantly reducing first-token latency. The projection is learned offline via simple ridge regression and operates on a compact summary of the full cache, making it efficient at inference time. We define a rigorous, reproducible evaluation protocol on small open-source models with pre-defined criteria for success: a significant reduction in latency must be achieved without a corresponding degradation in output quality, as measured by next-token logit divergence and task-specific metrics.

Introduction
- **Problem**: Autoregressive transformer inference requires an expensive prefill pass of complexity O(T²) over an input of length T to compute the key-value (KV) cache. This computation must be repeated for every new prompt, unless it shares an exact prefix with a previously processed prompt, creating a significant first-token latency bottleneck.
- **Hypothesis**: For semantically similar prompts, the functional effect of the KV cache on next-token generation lies in a low-dimensional subspace. Furthermore, we hypothesize that this subspace is locally linear, meaning the KV cache state of a target prompt can be approximated by a linear transformation of the cache state from a nearby source prompt.
- **Contribution**:
  1) A method to learn linear operators that project the KV cache from a source prompt to a target prompt via a low-dimensional summary.
  2) An inference pipeline that combines fast retrieval with cache projection to generate the first token with minimal latency, while initiating a background prefill pass to ensure fidelity for subsequent tokens.
  3) A reproducible evaluation protocol with pre-specified falsification criteria, designed for validation on small-scale open-source models.

Method
**Setting**
We consider a standard decoder-only transformer with L layers, H attention heads, and head dimension d. For a prefix of length T, the KV cache at layer ℓ consists of Kℓ ∈ R^{H×T×d} and Vℓ ∈ R^{H×T×d}. We introduce m compact summary slots per layer (e.g., m=16), which serve as a low-dimensional surrogate for the full cache.

**Offline Training**
We learn two sets of linear maps without fine-tuning the base model's weights.

1) **Cache-to-Summary Compression (C)**
   - **Objective**: To learn a per-layer, per-head linear map Cℓh that compresses a full KV cache into m summary slots. The compression is optimized to preserve the attention output for the first generated token.
   - **Data**: For a corpus of prompts D, we pre-compute the full cache (Kℓh(p), Vℓh(p)), the query for the first generated token Qℓh_next(p), and the true attention output Oℓh_next(p) for each prompt p.
   - **Learning**: We solve a ridge regression problem for each layer ℓ and head h. Let the full cache be concatenated as Xℓh(p) = [Kℓh(p) || Vℓh(p)] ∈ R^{T×2d}. The compressed summary Sℓh(p) = (K'ℓh, V'ℓh) ∈ R^{m×2d} is obtained by applying the map Cℓh. The loss minimizes the difference between the true attention output and the output using the compressed summary:
     ```
     argmin_{Cℓh} Σ_{p∈D} || Attn(Qℓh_next(p), K'ℓh(p), V'ℓh(p)) − Oℓh_next(p) ||² + λ||Cℓh||²
     ```
     where (K'ℓh(p), V'ℓh(p)) = Cℓh(Xℓh(p)).

2) **Cross-Prompt Summary Projector (M)**
   - **Objective**: To learn a per-layer, per-head linear map Mℓh that transforms the cache summary of a source prompt p_s into an approximation of the summary for a semantically similar target prompt p_t.
   - **Data**: A set of similar prompt pairs P = {(p_s, p_t)}, derived from paraphrase datasets. We pre-compute all summaries Sℓh(p) using the compression map C.
   - **Learning**: We solve a separate ridge regression problem for each Mℓh to minimize the projection error across all pairs in P:
     ```
     argmin_{Mℓh} Σ_{(p_s,p_t)∈P} || Mℓh Sℓh(p_s) − Sℓh(p_t) ||² + γ||Mℓh||²
     ```

**Inference**
- **Library**: We maintain an index (e.g., FAISS) of previously processed prompts, storing their embeddings and compressed cache summaries {Sℓ(p_i)}.
- **Execution**: Given a new prompt p_t:
  1) Retrieve the nearest neighbor p_s via embedding similarity. If similarity is below a threshold τ, fall back to a standard cold start.
  2) Load the source summary Sℓ(p_s) and compute the projected target summary: Ŝℓ(p_t) = Mℓ Sℓ(p_s).
  3) Initialize the `past_key_values` structure with the m-slot projected summary Ŝℓ(p_t). For models with absolute position embeddings, these slots occupy positions 0 to m−1. For RoPE models, they are assigned the corresponding initial rotary phases.
  4) Generate the first token immediately using this warm-started state.
  5) **(Optional) Background Refinement**: In a background thread, execute the full prefill pass for p_t. Once complete, seamlessly swap the projected cache with the exact cache for all subsequent token generations.

**Complexity and Integration**
- **Training**: Training consists of solving standard linear regression problems, which is fast and parallelizable.
- **Storage**: Each cached prompt requires storing only the summary, O(L×H×m×d), which is orders of magnitude smaller than the full cache.
- **Runtime**: First-token latency is reduced from a full forward pass to a vector retrieval and a set of matrix-vector products.

Experiments
**Evaluation Protocol**
Our experiments are designed as a falsification test for our central hypothesis.

- **Models**: GPT-2 (124M, 355M) and Pythia (160M, 410M) to test both learned absolute and Rotary Position Embeddings (RoPE).
- **Datasets**: Quora Question Pairs and PAWS for paraphrase-based evaluation. Templated QA and summarization prompts for instruction-following tasks.
- **Baselines**:
  - **Cold Start**: Standard prefill; the primary latency baseline.
  - **Exact Prefix Cache**: Upper bound on performance, lower bound on latency.
  - **Ablation 1 (No Projection)**: Use a retrieved source summary S(p_s) directly without projection.
  - **Ablation 2 (No Cache)**: Initialize only the final hidden state, demonstrating the necessity of the full KV cache state.
- **Metrics**:
  - **Latency**: Wall-clock time to first token.
  - **Distributional Fidelity**: KL divergence between the next-token probability distributions of the warm-started and cold-start models, KL(p_warm || p_cold).
  - **Task Quality**: Perplexity on held-out text and task-specific metrics (e.g., F1 for QA, ROUGE-L for summarization).
- **Success Criteria**: The hypothesis is supported if, for high-similarity prompts (e.g., cosine similarity > 0.9):
  1) First-token latency is reduced by >40% on average.
  2) Mean next-token KL divergence is ≤ 0.05.
  3) Task-specific quality metrics degrade by ≤ 1% absolute compared to the cold start.
  The hypothesis is falsified if these conditions are not met.

Discussion
Our method generalizes prompt caching from exact string matching to a semantic neighborhood, converting the O(T²) prefill computation into a fast retrieval and projection. This approach is conceptually related to prefix-tuning, but instead of learning a single static prefix per task, we dynamically generate a prompt-specific prefix via linear projection. The background cache refinement mechanism is critical, as it guarantees that any initial approximation error does not propagate through long generations.

**Limitations**
- **Domain Shift**: Projectors trained on one domain (e.g., paraphrases) may not generalize to another.
- **Retrieval Dependency**: The effectiveness of the method depends on a sufficiently dense library of past prompts to ensure good nearest neighbors are found.
- **Architectural Nuances**: The handling of position embeddings must be adapted for different model architectures. Our initial validation focuses on standard decoder models.
- **Approximation Fidelity**: While designed to be minimal, the projection introduces an approximation error that may not be acceptable for all use cases, though background refinement mitigates this.

Conclusion
We propose a method to warm-start LLM inference by linearly projecting KV cache summaries between semantically similar prompts. This technique replaces the expensive prefill pass with a fast retrieval and computation, offering a path to substantially lower first-token latency with a minimal and controllable impact on output quality. The method's simplicity, architectural non-invasiveness, and reliance on well-understood linear models make it a practical tool for accelerating LLM inference. We present a clear and reproducible protocol to test our hypothesis on publicly available models.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
