1) Quantum Rational Inattention Equilibrium (QRIE) in 2×2 Games
Summary: Players’ mixed strategies equal a logit quantal response whose temperature is an explicit function of qubit dephasing (T2) and circuit depth.
For a smart layperson: People don’t process all information perfectly; they make noisy choices. This theory links that “noise” to a physical noise you can dial on a quantum chip, predicting exactly how often each choice is made. If the chip gets noisier, people should act more randomly in a specific, measurable way.
Falsification: Implement 2-qubit circuits where each player’s action is RY(βΔu) with a tunable idle delay to induce dephasing; predict β ∝ effective coherence (from backend T2 and delay). In Python (Qiskit Aer), fit β from simulated counts; on IBM hardware, estimate β from 10k-shot counts and test βhardware/βsim vs predicted T2 scaling; reject if deviation > 15% across ≥3 delay settings.
IBM cost plan: Use IBM Quantum free tier or ≤5 circuits × 10k shots; under common pay-per-shot rates this is <$50, $0 on free queue. Python simulations (NumPy/Qiskit Aer) are free locally.
Novelty: It derives a closed-form link between physical decoherence and the logit parameter of quantal response, providing a directly testable bridge from information economics to device physics.

2) Entanglement-Selected Equilibria in Stag Hunt
Summary: The probability of coordination on the payoff-dominant equilibrium increases monotonically with entanglement fidelity and crosses the risk-dominant boundary at a calculable entanglement angle.
For a smart layperson: In coordination problems, groups sometimes settle on safe but worse outcomes. Shared quantum correlations (“entanglement”) can act like better advice, nudging groups toward the best outcome. The theory predicts exactly how much correlation is needed to flip behavior.
Falsification: Prepare a two-qubit partially entangled state cosθ|00⟩+sinθ|11⟩, apply local RZ(φi) to encode payoffs, measure Z to generate actions, and estimate P(CC). Python simulation yields the θ* where P(CC) surpasses the risk-dominant threshold; run on IBM at θ ∈ {0.1, 0.3, 0.5, 0.7} and verify the crossing within CIs; reject if no crossing appears with fidelity >0.9 predicted by calibration.
IBM cost plan: 4 θ-settings × 5k shots = 20k shots; <$20 on typical rates or $0 via free tier. Simulation via Qiskit Aer verifies the expected threshold before hardware.
Novelty: It provides a quantitative equilibrium-selection law mapping entanglement fidelity to the classic payoff-vs-risk dominance tradeoff in coordination games.

3) Helstrom-Limited Market Making
Summary: Optimal bid-ask spreads equal the minimum Bayesian loss implied by the Helstrom bound for discriminating informed vs liquidity trader states.
For a smart layperson: Market makers set spreads because they can’t perfectly tell who knows more. This theory treats that uncertainty like distinguishing two fuzzy quantum states and shows the spread should match the best possible error rate allowed by physics.
Falsification: Encode trader types as |ψ0⟩=|0⟩ and |ψ1⟩=cosφ|0⟩+sinφ|1⟩; implement the optimal single-qubit measurement (rotate by φ/2, measure Z) on IBM to estimate the empirical error ε̂(φ). In Python, compute the Helstrom error ε*(φ) and the implied optimal spread S*(ε*); compare S*(ε*) to simulated market losses using ε̂(φ); reject if spreads minimizing loss deviate from S*(ε̂) by >10% for φ ∈ {0.3,0.6,0.9}.
IBM cost plan: 3 φ-settings × 5k shots = 15k shots; <$15 paid or free via public queue. All payoff mapping and loss curves computed in Python.
Novelty: It unifies microstructure spreads with quantum state discrimination, yielding a precise, testable spread formula tied to physical limits of information.

4) Coherent-Bid Auctions Break Revenue Equivalence
Summary: Allowing coherent superpositions of discretized bids generically creates a revenue gap between first- and second-price auctions even under symmetric bidders.
For a smart layperson: Two classic auction types usually earn the seller the same money on average. If bidders can prepare “both bids at once” and let interference pick outcomes, that equal-earnings rule can break.
Falsification: With two qubits (one per bidder), encode bid choices {low, high} as RY(αi)|0⟩ and apply a controlled-phase to enable interference before measuring winners; implement first-price vs second-price allocation/pricing classically from outcomes. Python predicts a nonzero revenue gap ΔR(α, phase); on IBM, estimate ΔR with 10k shots per mechanism and reject if the sign and magnitude bands don’t match simulations within 95% CIs across 3 phase settings.
IBM cost plan: 2 mechanisms × 3 phases × 10k shots = 60k shots; typical cost <$60 or free on open devices. Python validation done locally.
Novelty: It isolates coherence (not entanglement) as a primitive that disrupts revenue equivalence under minimal, testable discretizations.

5) Quantum-Correlated Stable Matching Frontier
Summary: Quantum-correlated advice expands the ex-ante Pareto frontier of stable matches relative to classical correlated advice at a given fidelity cap.
For a smart layperson: In matching problems (like school placement), some pairings are better for everyone but are hard to coordinate on. Subtle quantum correlations can recommend pairings that make more people better off without causing blocking pairs.
Falsification: Implement a 2×2 toy matching with two qubits shared advice at tunable entanglement θ; map measurement outcomes to match recommendations and enforce stability filters in Python. IBM runs at θ ∈ {0,0.4,0.8} should yield average welfare exceeding the classical correlated bound predicted by simulation; reject if observed welfare never exceeds the classical envelope given measured state fidelities.
IBM cost plan: 3 θ-settings × 8k shots = 24k shots; <$25 paid or free. Python computes classical envelopes and stability checks.
Novelty: It formalizes and tests a quantum-correlated analogue of stable matching, quantifying welfare gains as a function of entanglement fidelity.

6) Entangled Public Goods Provision Threshold
Summary: In a binary-contribution public good, the contribution rate jumps at a calculable concurrence threshold when players receive entangled pre-play advice.
For a smart layperson: Public goods (like clean air) suffer from free-riding. If people share special correlated signals beforehand, contributions can rise sharply once the correlation is strong enough.
Falsification: Prepare a 3-qubit GHZ-like state cosθ|000⟩+sinθ|111⟩, apply local rotations encoding private cost signals, measure Z to decide contribute/withhold, and estimate contribution rates. Python determines the θ at which average payoff crosses the no-contribution equilibrium; IBM data at θ ∈ {0.2,0.5,0.8} should show the predicted jump; reject if no threshold behavior appears within error bars adjusted for measured GHZ fidelity.
IBM cost plan: 3 settings × 8k shots = 24k shots; <$25 or free. Simulation (Qiskit Aer) verifies robustness to realistic noise.
Novelty: It gives a sharp, fidelity-based phase-transition prediction for public good provision induced by quantum-correlated advice.

7) Decoherence-Linked Time Preference in Alternating-Offers Bargaining
Summary: The equilibrium split corresponds to a discount factor that is a deterministic function of dephasing over waiting intervals between offers.
For a smart layperson: In bargaining, patience determines who gets more. Here, “impatience” is modeled as loss of quantum coherence during waiting, yielding a formula for the final split.
Falsification: Represent each party’s “patience qubit,” insert idle delays Δt to induce dephasing before acceptance measurement (Z-threshold on RY(offer-dependent angle)); Python maps dephasing rate γ to an implied discount δ(γ) and split. On IBM, vary Δt over 3 values, estimate splits from 10k shots, and reject if inferred δ fails to track γ from calibration within 95% CIs.
IBM cost plan: 3 delays × 10k shots = 30k shots; <$30 or free. All mappings and thresholds coded in Python.
Novelty: It ties bargaining outcomes to physically measurable decoherence, yielding a directly testable “discount-from-decoherence” law.

8) Quantum Signaling Games with Non-Orthogonal Messages
Summary: Allowing senders to use non-orthogonal quantum messages enlarges the set of Perfect Bayesian equilibria via reduced distinguishability constraints.
For a smart layperson: In signaling (like education signaling ability), messages that are hard to perfectly read can change incentives. Quantum messages can be made deliberately hard to tell apart, altering which honest or deceptive patterns can survive.
Falsification: Implement a 1-qubit sender choosing |0⟩ or cosφ|0⟩+sinφ|1⟩ by type, and a receiver measurement rotated by θ; Python computes equilibrium regions in (φ,θ)-space. On IBM, estimate confusion matrices for φ ∈ {0.3,0.6} and test whether receiver best responses match predicted equilibria; reject if observed best responses contradict equilibrium inequalities derived from measured confusion rates.
IBM cost plan: 2 φ-settings × 6 θ-values × 3k shots ≈ 36k shots; <$40 or free. Python computes best-response maps and equilibrium checks.
Novelty: It operationalizes quantum indistinguishability as a strategic instrument, expanding empirically testable equilibrium sets in signaling.

9) Entry Deterrence via Coherent Commitment
Summary: Incumbents can credibly deter entry by preparing a commitment qubit whose interference visibility predicts the probability of carrying out a tough response.
For a smart layperson: A monopolist’s threat to fight entry is often not believable. Here, a quantum “commitment device” makes the threat’s credibility exactly equal to how sharp the interference pattern is.
Falsification: Prepare an incumbent qubit in superposition, apply a phase corresponding to “fight,” and measure in X to set response; visibility V sets the entry-deterrence threshold predicted by Python. IBM runs at V tuned via dephasing (vary idle time) should yield entrant’s observed entry rate matching the threshold; reject if entry frequencies don’t align with V within binomial CIs.
IBM cost plan: 3 visibility levels × 8k shots = 24k shots; <$25 or free. Python derives threshold policies and payoff comparisons.
Novelty: It provides a quantitative, physically grounded model of credible commitment in entry games via interference visibility.

10) Interference-Biased Mixed Strategies in Zero-Sum Games
Summary: Interference terms systematically bias mixed strategies away from classical Nash in discretized zero-sum games with coherent action encoding.
For a smart layperson: In pure competition games, players are supposed to randomize exactly. If choices are made coherently (allowing waves to interfere), the “randomization” skews in predictable ways.
Falsification: Encode a 2×2 zero-sum payoff with players’ choices as RY(αi)|0⟩, add a controlled-phase for interference, and measure Z; Python predicts bias in action frequencies as a function of phase. On IBM, sweep phase ∈ {0, π/4, π/2, 3π/4} with 8k shots each and reject if observed frequencies don’t follow the predicted sinusoidal bias pattern within RMSE tolerance <0.05.
IBM cost plan: 4 phases × 8k shots = 32k shots; <$35 or free on public devices. Python simulations (Qiskit Aer) benchmark expected counts with noise.
Novelty: It delivers a closed-form, testable interference law for deviations from classical mixed strategies in competitive games.
