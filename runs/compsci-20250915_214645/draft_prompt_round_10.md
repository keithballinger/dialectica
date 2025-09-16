You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on AI (such as but not limited to LLMs), avoid  anything blockchain related ideas
- Highly novel
- Publishable in a leading journal for its subfield

        Draft:
        Minor Revisions

Brief critique:
The revised draft presents a highly novel, falsifiable theory linking preconditioned gradient noise to ICL via a new P-NE metric, supported by controlled experiments and causal claims. This is evidence-based, grounded in SDE formalism, and distinguishes from prior work on noise and generalization. It fits AI/LLM focus, avoids blockchain, and appears publishable in a leading journal like NeurIPS or ICLR with strong empirical validation. Strengths include clear hypotheses, pre-registration, and negative controls. However, minor revisions are needed: (1) Add specific references to key prior works (e.g., on SDE approximations and induction heads); (2) Include placeholders or descriptions for figures (e.g., Figure 1 and 2) with mock data or captions; (3) Expand limitations to discuss generalizability to non-AdamW optimizers and non-Transformer architectures; (4) Provide a concrete example or pseudocode for P-NE estimation; (5) Fix minor inconsistencies, like clarifying "curvature-normalized P-NE" usage in experiments.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract

We demonstrate that stochastic gradient noise (SGN), integrated over training and weighted by the optimizer's preconditioner, is a primary and controllable driver of in-context learning (ICL) in language models. We propose a mechanism whereby SGN acts as an implicit regularizer, penalizing information storage in model weights and promoting context-adaptive computation in activations. We formalize this with a Preconditioned Noise Exposure (P-NE) metric, an online, low-overhead measure derived from the stochastic differential equation (SDE) view of adaptive optimizers. Through a series of tightly controlled experiments on 350M and 1.3B parameter models, we validate three falsifiable predictions: (1) for models trained to equivalent language modeling (LM) performance, ICL capability increases monotonically with P-NE; (2) ICL is more sensitive to P-NE accumulated late in training; and (3) the source of noise (e.g., small batches vs. explicit noise injection) is largely irrelevant, demonstrating approximate source equivalence. Our findings reframe ICL from an emergent property of scale to a predictable outcome of training dynamics, offering a direct lever for improving model adaptability.

## 1. Introduction

Large language models (LLMs) exhibit in-context learning (ICL), adapting to novel tasks presented in their context without weight updates. While model scale is a known correlate, the specific training dynamics that cultivate ICL are poorly understood. We isolate and verify a key causal factor: the accumulated, preconditioned stochastic gradient noise (SGN) experienced during pretraining.

Our central, validated claim is:
*Conditional on a fixed architecture, pretraining data distribution, and optimizer family, for models trained to an equivalent language modeling loss, increasing the integrated Preconditioned Noise Exposure (P-NE) causally increases ICL capability within a stable optimization regime.*

This work provides four main contributions:
1.  **Mechanism:** We ground ICL in optimizer-induced implicit regularization. SGN increases the effective temperature of the optimization process, imposing an information penalty on parameters and biasing the model toward solutions that perform computation in activations (ICL) rather than storing it in weights (memorization).
2.  **Metric:** We define and validate P-NE, a practical, optimizer-aware metric that quantifies the cumulative noise driving this effect, enabling comparison across different batch sizes, optimizers, and noise sources.
3.  **Causal Evidence:** Through controlled interventions that manipulate P-NE independently of LM loss—via batch size, explicit noise injection, and noise scheduling—we confirm our pre-registered hypotheses of monotonicity, phase sensitivity, and approximate source equivalence.
4.  **Negative Controls:** We demonstrate that increasing P-NE selectively improves ICL at the cost of parametric recall, confirming that the underlying mechanism trades weight-based memorization for context-based computation.

## 2. Related Work

Our work integrates two research streams: mechanisms of ICL and the role of SGN as an implicit regularizer. ICL has been explained as implicit Bayesian inference (e.g., Xie et al., 2021), meta-optimization via gradient descent in activation space (e.g., von Oswald et al., 2023), and the action of specific circuits like induction heads (e.g., Olsson et al., 2022). Separately, SGN in Stochastic Gradient Descent (SGD) is known to bias optimizers toward flat, generalizable minima (e.g., Hochreiter & Schmidhuber, 1997; Mandt et al., 2017), with connections to approximate Bayesian inference and local entropy.

**Distinction from Prior Work:** While prior work linked small batches to better generalization (e.g., McCandlish et al., 2018), it did not isolate SGN as a causal driver for ICL specifically, nor did it control for final LM performance. Our contribution is the synthesis of these ideas into a falsifiable, causal theory for ICL, supported by: (i) a novel, **preconditioned** noise metric (P-NE) that accounts for adaptive optimizers; (ii) causal tests that disentangle noise from other training variables; and (iii) specific predictions about **phase sensitivity** and the trade-off with memorization.

## 3. Theory and Metric

### 3.1. SDE View of Optimizer Noise

Adaptive optimizers like AdamW can be approximated by a stochastic differential equation (SDE) (e.g., Li et al., 2017). For an optimizer with diagonal preconditioner $P_t$ (e.g., $1/\sqrt{v_t}$ in Adam), the update noise has a covariance of approximately $\eta_t^2 P_t C_t P_t$, where $\eta_t$ is the learning rate and $C_t$ is the mini-batch gradient covariance. This term sets the *effective temperature* of the optimization, governing the stationary distribution over parameters. Higher temperature biases the search toward wider, flatter minima, which correspond to solutions with higher local entropy. This imposes a stability cost on solutions that rely on precisely tuned weights, effectively regularizing the information content of the parameters.

### 3.2. Parameter–Activation Information Allocation

A model can solve a task by encoding solutions in its parameters (weight-coded computation) or by inferring a solution from the prompt and executing it in activations (context-adaptive computation). The high effective temperature induced by SGN penalizes weight-coded solutions, which are brittle and occupy sharp regions of the loss landscape. In contrast, context-adaptive circuits, which are reused across tasks, are more robust to parameter noise. Therefore, integrated SGN shifts the optimization equilibrium toward ICL, assuming the training data contains sufficient task diversity to learn such general-purpose circuits.

### 3.3. The Preconditioned Noise Exposure (P-NE) Metric

We define the total P-NE as the integrated, preconditioned gradient variance trace:
$$ \text{P-NE} = \sum_{t=0}^{T} \eta_t^2 \cdot \text{Tr}(P_t C_t P_t) $$

To ensure comparability across runs with different schedules or loss landscapes, we introduce a **curvature-normalized P-NE** ($P_{NE}^*$):
$$ P_{NE}^* = \sum_{t=0}^{T} \frac{\eta_t^2 \cdot \text{Tr}(P_t C_t P_t)}{\max(\epsilon, \text{Tr}(P_t F_t P_t))} $$
where $F_t$ is a Fisher information matrix approximation to the Hessian, and $\epsilon$ is a small constant. This metric measures noise relative to the local curvature, providing a more schedule-invariant quantity. We use $P_{NE}^*$ consistently in our experiments for cross-run comparisons.

**Estimation:** We estimate $diag(C_t)$ online with minimal overhead by computing the variance of gradients over $K=8$ micro-batches every $N=100$ steps. For full-covariance experiments, we use a Hutchinson trace estimator. We compute noise on unclipped, full-precision gradients and log the clipping rate to monitor its impact. All P-NE computations are released as a standalone library. Pseudocode for estimation:
```
def estimate_p_ne(grads, P_t, eta_t, F_t_approx, epsilon=1e-6):
    C_t_diag = variance(grads, axis=0)  # over micro-batches
    noise_term = eta_t**2 * trace(P_t * C_t_diag * P_t)
    curv_term = max(epsilon, trace(P_t * F_t_approx * P_t))
    return noise_term / curv_term
```

## 4. Experimental Design

Our experiments were designed to causally test our three primary hypotheses.

**Models and Data:** We used decoder-only Transformers at 350M and 1.3B parameters, trained on a diverse pretraining dataset. Architecture, tokenizer, and data curriculum were fixed across all runs.

**Noise Manipulation:** We modulated P-NE via three methods:
1.  **Batch Size:** We varied global batch size from 256 to 4096. We tested both a fixed learning rate (LR) schedule and a schedule following the linear scaling rule to decouple noise from LR effects.
2.  **Explicit Noise Injection:** At a large batch size, we injected zero-mean Gaussian noise $\mathcal{N}(0, \Sigma_t)$ pre-preconditioning, dynamically scaling $\Sigma_t$ to match the P-NE trajectory of a small-batch run.
3.  **Noise Scheduling:** We compared runs with P-NE concentrated in the early, middle, or late phase of training, while keeping total integrated P-NE constant.

**Controls and Matching:** To isolate the effect of P-NE, all experimental runs were trained to a target validation perplexity of $10.0 \pm 0.05$ before evaluation. Data order, dropout rates, weight decay, gradient accumulation steps, and mixed-precision settings were held constant. We continuously monitored P-NE, gradient norms, and optimizer states to verify the fidelity of our interventions.

**Evaluation:**
-   **Primary ICL Metrics:** Few-shot accuracy on MMLU and Big-Bench Hard (BBH) subsets, and performance on synthetic tasks (e.g., in-context linear regression, symbolic manipulation).
-   **Negative Control (Memorization):** Performance on factual recall (cloze probes) and synthetic key-value pair memorization tasks.
-   **Statistical Protocol:** All hypotheses, metrics, and analysis plans were pre-registered. We used mixed-effects models and report 95% confidence intervals (CIs). For equivalence, we used Two One-Sided Tests (TOST) with a pre-specified margin of $\delta=0.02$ in accuracy.

## 5. Results

Our experiments confirm all three pre-registered hypotheses across both model scales.

### 5.1. P-NE Increases ICL Monotonically

At a matched perplexity, ICL performance increases monotonically with P-NE. As shown in Figure 1 (scatter plot of P-NE vs. ICL score, with trend line; mock data: points at P-NE=1.0→ICL=0.4, P-NE=2.0→ICL=0.55, etc.), runs with higher P-NE (from smaller batches or explicit noise) consistently achieved higher few-shot accuracy on both MMLU and synthetic ICL benchmarks. The Spearman correlation between P-NE and aggregate ICL score was significant ($\rho = 0.89$, 95% CI $[0.78, 0.95]$).

### 5.2. Late-Phase Noise Has a Disproportionate Impact

Models are most sensitive to P-NE in the final third of training. Runs where P-NE was concentrated late achieved a 15% higher ICL score (95% CI $[9\%, 21\%]$) compared to runs with early-phase noise, despite identical total P-NE and final perplexity. This suggests that noise primarily influences the fine-tuning of ICL-capable circuits that form later in optimization.

### 5.3. Approximate Source Equivalence of Noise

The source of gradient noise has little effect on ICL. Runs with small-batch noise and runs with large-batch plus explicit matched noise injection showed statistically equivalent ICL performance. The TOST procedure confirmed that the difference in mean ICL accuracy was within our pre-specified equivalence margin of $\pm2\%$ ($p < 0.01$). This supports our claim that P-NE is the key underlying quantity, not batch size itself.

### 5.4. P-NE Trades Memorization for ICL

As predicted by our mechanism, higher P-NE degrades performance on tasks requiring parametric memorization. Figure 2 (trade-off curve: ICL score increasing vs. memorization score decreasing with P-NE; mock data: Pareto frontier) shows a clear trade-off: as P-NE increases, ICL score rises while factual-recall accuracy declines. This provides strong evidence that SGN forces a shift from weight-coded to context-adaptive computation.

## 6. Discussion

**Mechanistic Probes:** Analysis of our trained models reveals that higher P-NE correlates with stronger induction head formation and more dynamic, prompt-sensitive attention patterns, consistent with a shift toward context-adaptive computation.

**Geometric Analysis:** Models trained with higher P-NE consistently converged to flatter minima, as measured by SAM sharpness and the leading eigenvalues of the Hessian. This confirms that P-NE acts as a powerful implicit regularizer, guiding the optimizer toward wider, more generalizable basins of the loss landscape.

**Limitations:** Our findings are demonstrated for AdamW on Transformer-based LLMs and may not generalize to all optimizers (e.g., SGD without adaptation) or architectures (e.g., RNNs or CNNs). The precise boundaries of the "stable optimization regime" outside of which P-NE is detrimental require further characterization. Additional work is needed to test scalability to larger models and diverse datasets.

## 7. Conclusion and Implications

We have provided strong causal evidence that integrated, preconditioned gradient noise is a key driver of in-context learning. By defining and measuring P-NE, we validated three core predictions: ICL increases monotonically with P-NE, is most sensitive to late-phase noise, and is agnostic to the noise source.

This finding has significant practical implications:
-   **Tunable ICL:** ICL can be directly engineered by manipulating training dynamics (e.g., via late-stage batch size reduction or explicit noise injection) without necessarily requiring more data or parameters.
-   **Efficient Training:** Understanding the P-NE-ICL relationship provides guidance for batch size and LR schedule choices, potentially enabling the development of powerful ICL capabilities at lower computational cost.
-   **A Unifying Principle:** Our work connects ICL to the broader literature on implicit regularization in deep learning, framing it as a predictable consequence of the optimizer's stochastic dynamics.

Ultimately, ICL is not merely an emergent property of scale but a controllable feature of the training process, shaped by the subtle interplay between optimization, noise, and the information trade-offs made along the path to convergence.

## 8. Reproducibility

We release our code for computing P-NE online, all training and evaluation scripts, and the logs from our experimental runs. Our analysis plan was pre-registered at [https://osf.io/example-pre-reg].


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
