1) Entropy-Triggered Compute Regimes in LLMs
Summary: LLMs switch between distinct internal computation modes when next-token predictive entropy crosses stable thresholds.
For a smart layperson: When the model is very unsure about the next word, it may internally change how it “thinks,” using different parts of its circuitry. This hypothesis says those switches occur at clear uncertainty levels, not gradually. Finding such switches would explain why models sometimes suddenly become more verbose or cautious.
Falsification: Sweep logit scaling to vary entropy on fixed prompts and record per-layer activation sparsity/attention-head usage; test for change-point structure aligned to entropy thresholds using change-point detection. If no consistent thresholds appear across prompts/models (e.g., Pythia-1.4B, Llama-2-7B), reject. Verify invariance under sampling vs greedy decoding.
Novelty: Prior work measures uncertainty but does not posit discrete, entropy-triggered regime shifts in internal compute during inference.

2) KL-Acceptance Law for Speculative Decoding
Summary: The acceptance rate in speculative decoding is approximately exp(−KL(target||draft)) under standard acceptance rules.
For a smart layperson: Fast “draft” models propose words and the big model approves or revises them; how often approvals happen controls speed. This theory predicts a precise mathematical link between approval rate and how different the two models’ beliefs are. If true, we can pick draft models to hit target speeds reliably.
Falsification: For multiple draft/target pairs (e.g., Phi-2→Mistral-7B, Pythia-1.4B→Llama-2-7B), compute tokenwise KL on shared prompts and measure empirical acceptance rates; test exponential relation via regression and goodness-of-fit. If the relation fails systematically or requires ad-hoc corrections beyond a constant factor, reject.
Novelty: No published work provides a simple predictive law tying speculative acceptance rate directly to tokenwise KL divergence.

3) Universal Low-Rank Manifold of KV Caches
Summary: During inference, key/value caches lie on a low-rank manifold whose dimension is nearly constant across tasks and prompts.
For a smart layperson: The model stores past information in large memory tensors; this claims that those big memories actually live on a much smaller shape. If true, we can compress memory a lot without losing accuracy, speeding up long-context use.
Falsification: Collect KV tensors across datasets (QA, code, stories) for small models; perform per-layer SVD and estimate intrinsic dimension via participation ratio/p-airgap metrics; test if a fixed low rank explains >95% variance across tasks and contexts. If required rank varies widely or is high, reject.
Novelty: KV compression exists, but a task-agnostic, near-constant-dimensional manifold hypothesis has not been articulated or tested.

4) Consensus-Gated Early Exit Improves Speed Without Loss
Summary: Agreement among k perturbed decoders (e.g., temperatures) predicts that deeper layers add negligible accuracy, enabling safe early exit.
For a smart layperson: If several slightly different decoding tries all say the same next word, we can stop the model early and still be right. This would cut compute while keeping quality. It’s like ending a vote early when everyone already agrees.
Falsification: Implement per-token early exit when top-1 tokens across k temperatures agree above a threshold; compare latency vs accuracy on benchmarks (WikiText, SQuAD, GSM8K small) to fixed-depth baselines. If accuracy drops beyond a small margin at matched speed or consensus isn’t predictive, reject.
Novelty: Prior early-exit methods rely on internal confidence; using agreement among cheap perturbations as a sufficiency test is new.

5) Geodesic Smoothness of Logit Trajectories Predicts Coherence
Summary: Coherent generations follow low-curvature geodesics in the probability simplex, while hallucinations correlate with curvature spikes.
For a smart layperson: As the model writes, its belief over next words traces a path; smooth, gentle paths yield sensible text, while jerky turns signal going off-track. Measuring the “bend” of this path could warn about upcoming errors.
Falsification: Compute per-step Fisher-Rao geodesic curvature of logits during generation and correlate with human/automatic coherence and hallucination labels (TruthfulQA, FactCC). If curvature doesn’t predict errors better than entropy/surprisal baselines, reject.
Novelty: Linking differential-geometric properties of logit trajectories to generation quality has not been explored.

6) Beam Width >2 Degrades Factuality in Instruction LLMs
Summary: For instruction-tuned models, increasing beam width beyond 2 reduces factual accuracy due to mode bias toward generic templates.
For a smart layperson: Searching more candidate sentences seems good, but it can push the model into safe-sounding, generic answers that are less correct. This claims that beyond a small beam, answers get more templated and less factual.
Falsification: Evaluate factual QA (Natural Questions, TriviaQA subsets) with beam widths {1,2,4,8} on small instruction models; measure exact match and hallucination rate. If wider beams don’t reduce factuality relative to width 1–2 at matched length, reject.
Novelty: While beam search issues are known, a systematic, falsifiable claim about a specific width threshold and factuality in instruction models is absent.

7) Topological Motifs in Attention Graphs Encode Task Type
Summary: Persistent homology features of attention graphs reliably classify task type and predict task-specific head usage.
For a smart layperson: We can turn attention patterns into shapes and holes, then read off what kind of task the model is doing from those shapes. If the shapes match tasks, it reveals hidden structure in how the model routes information.
Falsification: Build token-level attention graphs per layer; compute persistence diagrams and vectorize; train a simple classifier to predict task labels (translation, QA, code) across prompts and models; test generalization across datasets. Failure to beat strong baselines or lack of consistent motifs rejects the theory.
Novelty: Using topological data analysis to derive invariant attention motifs tied to task identity is new.

8) Mutual Information Plateau Limits Long-Context Gains
Summary: When cross-token mutual information beyond distance d plateaus, extending context length past d yields negligible perplexity gains for that model scale.
For a smart layperson: If the model stops picking up useful connections after a certain distance, giving it even longer history won’t help much. Measuring this “information reach” tells us when longer context is wasted.
Falsification: Estimate distance-conditioned MI via contrastive estimators (InfoNCE) on long texts and relate the plateau point to perplexity improvements from context extensions on the same data; if extensions help despite a plateau, reject.
Novelty: Prior context studies report empirical gains but do not ground them in an information-theoretic plateau criterion per model scale.

9) Clause-Reordering Invariance of KV Subspaces
Summary: Reordering independent clauses leaves the later-layer KV cache subspace invariant up to a low-distortion rotation.
For a smart layperson: If we shuffle parts of a sentence that don’t depend on each other, the model’s internal memory should essentially stay the same shape. This would show the model separates meaning from word order when it can.
Falsification: Generate paired prompts with clause reorders preserving semantics; extract per-layer KV, compute subspace principal angles/Procrustes distances; test for low distortion in later layers compared to early layers and to paraphrase controls. If invariance fails, reject.
Novelty: Invariance in KV subspaces under controlled syntactic reorderings has not been characterized.

10) Monotone Position-Dependent Temperature Improves Truthfulness
Summary: A monotonically decreasing temperature schedule over positions increases truthfulness at equal average perplexity.
For a smart layperson: Starting a bit creative and getting more cautious as the answer unfolds can make models more truthful without making them dull overall. The trick is to cool down steadily as more context accumulates.
Falsification: Construct per-position temperature schedules with equal average perplexity to a constant-temperature baseline; compare truthfulness (TruthfulQA, BioASQ subsets) and hallucination metrics on small models. If no improvement appears across reasonable schedules, reject.
Novelty: Decoding work tunes a single temperature; a principled, monotone per-position schedule tied to truthfulness at fixed perplexity is new.
