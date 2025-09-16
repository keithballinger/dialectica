1) Entropic Lower Bounds on Branch Prediction Accuracy
Summary: The minimum achievable misprediction rate for any branch predictor on a program is lower-bounded by the branch outcome stream’s entropy rate, independent of microarchitecture.
For a smart layperson: Modern CPUs guess the results of “if” statements to go faster, but some sequences are inherently unpredictable. This theory says we can measure how unpredictable a program’s branches are, and that no hardware can beat the limit implied by that measured unpredictability.
Falsification: Collect branch outcome traces from diverse programs, estimate entropy rate with context-tree weighting or similar, and build state-of-the-art predictors; if any predictor consistently beats the lower bound across traces by more than statistical error, the theory is false. Repeat across different CPUs to test microarchitecture independence; if bounds vary beyond measurement error solely due to hardware, the theory is false.
Novelty: Establishes a quantitative, architecture-invariant lower bound tied to empirically estimable entropy rate, rather than qualitative or hardware-specific limits.

2) Information-Theoretic Model of Code Comprehension
Summary: Programmer comprehension time scales linearly with the Shannon entropy of a function’s control- and data-flow graph, independent of programming language syntax and identifier names.
For a smart layperson: Understanding code gets harder the more “surprise” there is in how control and data move through it. This theory claims a single number measuring that surprise predicts how long people take to understand the code, regardless of surface style.
Falsification: Construct code snippets with matched functionality but varying graph entropy, translate them across multiple languages with different naming schemes, and measure comprehension time via tasks and eye-tracking; if entropy does not linearly predict time or language/naming effects dominate, the theory is false.
Novelty: Introduces an explicit, testable information-theoretic law for human code understanding rather than empirical heuristics or readability proxies.

3) Manifold-Dimension Law for Generalization across Architectures
Summary: For supervised deep networks trained by SGD, test error curves collapse when plotted against sample size divided by the data’s effective manifold dimension, regardless of architecture and width.
For a smart layperson: Many learning problems live on a lower-dimensional shape inside high-dimensional data; the claim is that how well different neural networks generalize depends mainly on how much data you have relative to that hidden dimension. If true, very different model types will follow the same learning curve once you account for that dimension.
Falsification: Estimate effective dimension on standard datasets using multiple estimators, train diverse architectures (CNNs, Transformers, MLPs) across scales, and test whether test-error vs. N/d_eff curves align within predefined tolerance; persistent non-alignment across estimators or datasets falsifies the theory.
Novelty: Predicts an architecture-invariant scaling collapse tied to intrinsic data geometry, extending beyond known model-specific scaling laws.

4) Capacity Law for Cache Side-Channel Attacks
Summary: Given a fixed leakage model with secret-dependent memory accesses, the key-recovery success rate of last-level cache timing attacks is determined by an empirically measurable channel capacity that depends only on noise and eviction dynamics.
For a smart layperson: Attacks that read secrets through timing act like sending messages over a noisy line; this theory says the attack’s success depends only on how much information that line can carry, which we can measure, not on idiosyncrasies of the victim code.
Falsification: On multiple hardware platforms, measure cache side-channel capacity via stimulus–response experiments, then run standard key-recovery attacks across diverse victim implementations with the same leakage model; if success rates systematically exceed or fall short of what the measured capacity predicts, the theory is false.
Novelty: Provides a quantitative mapping from measured microarchitectural capacity to attack success, replacing qualitative vulnerability assessments.

5) Spectral-Gap Law for Tail Latency in RPC Services
Summary: In microservice graphs with bounded service-time variability, the 99th-percentile end-to-end latency scales inversely with the spectral gap of the service dependency graph rather than with path length.
For a smart layperson: In complex service meshes, slowdowns come from how the services are connected, not just how many steps a request takes. This theory says a single number describing the graph’s “connectedness” predicts tail latency.
Falsification: Build microservice topologies with controlled spectral gaps (via added edges/sharding), fix per-node service-time distributions, and measure p99 latency under load; if p99 does not scale with inverse spectral gap or depends primarily on path length when variability is bounded, the theory is false.
Novelty: Links tail latency to a concrete spectral property of the service graph, moving beyond series/parallel or hop-count reasoning.

6) Equivalence of Test-Time Training and Adaptive Kernel Regression
Summary: For models with a frozen feature extractor and trainable last layer, small-step test-time training is equivalent to kernel regression with input-dependent bandwidth set by the local Fisher information.
For a smart layperson: When models lightly tune themselves on each new test example, they behave like a smoother that adapts how much to blur based on local confidence. This theory makes that intuition precise and predicts their outputs.
Falsification: On synthetic and real datasets, compare predictions and calibration of last-layer-adapted models under test-time training to those of an adaptive kernel with bandwidth derived from Fisher information; significant, systematic deviations falsify the theory.
Novelty: Unifies disparate adaptation methods under a single, testable statistical mechanism with explicit predictive equivalence.

7) Percolation Threshold Governs In-Context Learning in Transformers
Summary: Strong in-context learning emerges when the per-layer attention graph exceeds a percolation threshold that guarantees low-stretch paths from prompt tokens to targets.
For a smart layperson: Transformers learn from the prompt by passing information along attention links; this theory says there’s a tipping point in how connected those links must be for the model to use examples effectively.
Falsification: Train families of sparse-attention Transformers with controlled sparsity patterns, measure per-layer percolation and path stretch, and evaluate in-context learning tasks; absence of threshold-like performance changes at predicted connectivity levels falsifies the theory.
Novelty: Explains in-context learning as a graph phase transition, replacing architecture-specific heuristics with a structural criterion.

8) Pathwidth Bound on Reverse-Mode AD Memory
Summary: The minimum peak memory required by reverse-mode automatic differentiation equals the computation graph’s pathwidth up to a constant factor.
For a smart layperson: Backpropagation must remember certain intermediate values; this theory ties the best possible memory use to a well-studied graph property that measures how “linearizable” the computation is.
Falsification: For benchmark graphs (chains, trees, grids, small real models), compute exact/approximate pathwidth, implement optimal checkpointing schedules, and compare peak memory; consistent deviations beyond a constant-factor envelope falsify the theory.
Novelty: Connects AD memory optimality to a canonical graph parameter, yielding precise, verifiable bounds across models.

9) Grammar-Prior Entropy Governs Zero-Shot Program Synthesis
Summary: The zero-shot success rate of code language models on a target language is predicted by the entropy of the grammar prior induced from training, largely independent of tokenization scheme.
For a smart layperson: How well a model writes code in a new language depends on how complex it believes the language’s grammar is; lower assumed complexity means better zero-shot performance.
Falsification: Induce probabilistic grammars from training corpora, estimate their entropies for multiple target languages, and correlate with zero-shot synthesis success while varying tokenization; weak or tokenization-dependent correlations falsify the theory.
Novelty: Grounds cross-language code generation in formal-language entropy rather than surface similarity or dataset size alone.

10) Representation Controllability Curvature Law
Summary: The fraction of features that can be reliably steered by sparse linear interventions in a model’s representation is determined by the average sectional curvature of the representation manifold.
For a smart layperson: A model’s internal space may be flatter or more curved; this theory says flatter spaces allow more features to be nudged with simple controls, while curved spaces resist.
Falsification: Estimate Riemannian curvature from embeddings (e.g., via pullback metrics), apply sparse steering vectors to induce target changes, and test whether the steerable fraction aligns with curvature predictions across architectures and datasets; systematic mismatches falsify the theory.
Novelty: Introduces differential-geometric control as a predictive framework for steerability in learned representations with measurable quantities.
