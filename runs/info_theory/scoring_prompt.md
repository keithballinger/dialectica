You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Rate each idea on a scale of 1–10 for novelty, falsifiability, and feasibility under the constraints. Provide a one-sentence rationale per idea.

        Constraints of Paper:
        From: constraints/info_theory.md

- In the field of Information Theory
- Highly novel
- Publishable in a leading journal for its subfield

        Ideas:
        1) Universal Finite-Blocklength Law for Learned Channel Codes
Summary: Any end-to-end trained neural code over a memoryless channel obeys the normal approximation with dispersion equal to the channel’s information density dispersion up to a constant representation penalty that vanishes with training data and model size.
For a smart layperson: Modern error-correcting codes are now often learned by neural networks instead of hand-designed. This theory says that, when trained well, such learned codes behave like the best classical codes even for short messages, with a simple formula predicting error rates. The only extra cost is a small penalty that shrinks as models and datasets grow.
Falsification: Train diverse neural encoders/decoders across canonical channels (AWGN, BSC, Rayleigh) and blocklengths; estimate empirical error vs rate and fit the normal approximation. If the fitted dispersion systematically deviates from the channel dispersion by more than a diminishing penalty as model/data scale, the theory is falsified.
Novelty: It proposes a universal finite-blocklength law for learned (not hand-crafted) codes with a specific, vanishing representation penalty.

2) The Exact Privacy–Capacity Frontier under (ε,δ)-Differential Privacy
Summary: The maximum mutual information between a dataset and a privatized output under (ε,δ)-differential privacy equals a closed-form function g(ε,δ) independent of the data distribution and is achieved by staircase mechanisms.
For a smart layperson: Differential privacy limits how much information about any individual leaks. This theory gives the exact best-possible information transfer under a given privacy level, not just a bound. It also identifies a simple family of mechanisms that achieve this limit.
Falsification: Construct any (ε,δ)-DP mechanism and empirically/analytically estimate mutual information for worst-case priors; if MI exceeds g(ε,δ) for any ε,δ, the theory is false, or if staircase mechanisms fail to approach g(ε,δ), it is refuted.
Novelty: It claims an exact, prior-free privacy–information curve and its achievers, surpassing existing upper/lower bounds.

3) Predictive Work Principle for Semantic Compression
Summary: The minimal thermodynamic work needed to compress a time series into a representation that preserves a fixed prediction accuracy equals kT ln 2 times the predictive information captured by that representation.
For a smart layperson: Compressing data you’ll use to predict the future throws away some bits; physics says erasing bits costs energy. This theory links the energy cost directly to how much predictive content you keep, making “useful information” a physical resource.
Falsification: Implement controlled bit-erasure compressors on time-series sources (e.g., Markov, AR) using nanoscale memory arrays; measure work to achieve fixed predictive accuracy and compare to predictive mutual information estimates. Systematic violations beyond experimental error falsify the principle.
Novelty: It bridges predictive information from information theory with measurable thermodynamic work for task-relevant (semantic) compression.

4) Graph-Spectral Law for Information Contraction in Structured Channels
Summary: For any memoryless channel whose zero pattern matches a fixed bipartite graph, the strong data-processing coefficient for mutual information equals the square of the second-largest singular value of the graph’s normalized adjacency.
For a smart layperson: Some channels never confuse certain symbols, a structure you can draw as a graph. This theory says how much information such a channel must lose is encoded in a simple spectral number of that graph.
Falsification: Generate families of channels with identical zero patterns but varying nonzero weights; estimate contraction coefficients via convex programming or coupling methods and compare against the predicted spectral value. A single counterexample with mismatch falsifies the claim.
Novelty: It ties a core contraction property directly to a graph-spectrum invariant for a broad class of structured channels.

5) Directed-Information Lower Bound on Generalization in Stochastic Learning
Summary: The expected generalization gap of any stochastic training algorithm is bounded below by a constant multiple of the directed information from the training data sequence to the learned parameters.
For a smart layperson: How much a model overfits depends on how much the data actually influences its parameters during training. This theory quantifies that influence precisely and says you cannot generalize better than this information flow allows.
Falsification: Train models with controllable noise schedules on synthetic and real datasets; estimate directed information via pathwise likelihood ratios or variational bounds and compare to measured generalization gaps. If observed gaps routinely fall below the bound beyond estimation error, the theory is false.
Novelty: Prior work gives upper bounds via mutual information; this provides a universal lower bound using directed information that depends on training dynamics.

6) Superadditivity Criterion for Classical Finite-State Channels with Correlated Encoding
Summary: For finite-state channels with deterministic state evolution, per-symbol capacity is strictly superadditive in blocklength if and only if the state graph contains at least two disjoint cycles with distinct single-cycle capacities and the encoder uses pre-shared randomness to correlate inputs across time.
For a smart layperson: Some channels have memory, like a device that behaves differently depending on its internal state. This theory predicts exactly when sending longer, correlated messages gives you a bigger per-symbol payoff than sending short, independent ones.
Falsification: Construct finite-state channels meeting and violating the cycle condition; compute achievable rates with correlated coding via dynamic programming and compare to single-letter capacities. Any counterexample to the “if and only if” condition falsifies the theory.
Novelty: It gives a sharp structural characterization for when classical channels exhibit superadditivity using correlated encoders.

7) Algorithmic Typicality Bound Linking Kolmogorov Complexity and Information Density
Summary: For any stationary ergodic source, sequences with Kolmogorov complexity k bits below n times the entropy have information density tails at least 2^{-k} with universal constants, yielding a finite-block bound on atypicality.
For a smart layperson: “Unusually simple” data strings are rare, and they also look statistically surprising. This theory precisely links being algorithmically simple to being statistically surprising in a way that works for finite lengths.
Falsification: Use universal compressors (e.g., Lempel–Ziv) as proxies for complexity on synthetic ergodic sources; estimate empirical information density and tail probabilities. Systematic violations of the predicted tail lower bounds for growing n falsify the claim.
Novelty: It provides a quantitative, distribution-agnostic bridge between algorithmic complexity deficits and information-spectrum tails.

8) An Operationally Unique Partial Information Decomposition via Optimal Guessing
Summary: The mutual information between two sources and a target admits a unique nonnegative decomposition into redundant, unique, and synergistic parts defined by changes in optimal guessing performance under controlled side-information.
For a smart layperson: When two signals inform a target, some bits are shared, some are unique, and some only matter together. This theory defines these parts by how much they help you guess correctly, yielding a single, unambiguous split.
Falsification: Construct canonical distributions (e.g., copy, XOR, AND, noisy variants) and compute the proposed decomposition; check axioms (symmetry, self-redundancy, data processing) and compare against operational tasks. Any violation of axioms or nonuniqueness across equivalent formulations falsifies it.
Novelty: It offers a single operational definition that claims uniqueness and satisfies core axioms long debated in partial information decomposition.

9) Multi-View 1/r Law for Task-Oriented Rate–Distortion
Summary: For conditionally independent views given a latent variable, the minimal per-view rate to achieve fixed task risk scales as 1/r of the single-view rate as the number of views r grows, up to a vanishing remainder.
For a smart layperson: If many sensors see the same hidden scene independently, each sensor can send less information as you add more sensors. This theory predicts a simple one-over-r saving in how much each must communicate to keep task performance constant.
Falsification: Simulate latent-variable generative models and real multi-view datasets; train bottlenecked encoders with a fixed downstream task and measure achieved risk vs total and per-view rates. Deviations from 1/r scaling beyond finite-sample effects falsify the law.
Novelty: It gives an operational, task-level rate–distortion scaling that is universal across models satisfying conditional independence.

10) Minimax-Optimal Estimability Frontier for Variational f-Divergence Estimators
Summary: Variational neural estimators of f-divergences achieve minimax-optimal mean-squared error if and only if the feature operator has bounded spectral norm, yielding a sample complexity proportional to the inverse chi-squared affinity between distributions.
For a smart layperson: Neural methods estimate how different two distributions are, but when are they statistically optimal? This theory gives a sharp criterion that predicts when these estimators are as good as possible and how many samples they need.
Falsification: Create synthetic distribution pairs with controlled spectrum of the feature operator; train variational estimators and measure estimation error vs sample size. If optimal rates occur without the spectral condition or fail to occur with it, the theory is falsified.
Novelty: It links minimax optimality of popular neural information estimators to a precise spectral condition and yields explicit sample-complexity scaling.


        Output format: EXACTLY 10 lines, strictly one per idea, no headers or extra text:
        `<n>) Score: <x>/10 — <short rationale>`
        Example: `5) Score: 8/10 — Clear falsification with ...`
