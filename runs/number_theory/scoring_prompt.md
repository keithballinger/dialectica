You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Rate each idea on a scale of 1–10 for novelty, falsifiability, and feasibility under the constraints. Provide a one-sentence rationale per idea.

        Constraints of Paper:
        From: constraints/number_theory.md

- Related to Automorphic Forms & Functoriality
- Highly novel
- Publishable in a leading journal for its subfield
- Uses simulations in python for validation

        Ideas:
        1) Convolutional Sato–Tate for Tensor Products
Summary: For non-CM GL(2) newforms f and g, the normalized prime coefficients b_p = a_p(f)a_p(g)/(2√p)^2 have empirical distribution converging to the pushforward of Haar measure on SU(2)×SU(2) under the product of characters.
For a smart layperson: Each prime p contributes a pair of “angles” encoding f and g; multiplying their signals gives a new signal. The theory says that, across many primes, this product behaves like two independent idealized rotations combined, with a precise predicted distribution. If true, it directly witnesses the tensor-product functoriality at the level of prime data.
Falsification: Use Sage/Python to gather Hecke eigenvalues for many non-CM newforms and form b_p up to p ≤ X. Compare the empirical density to the explicit convolutional Sato–Tate density via KS or Wasserstein tests; repeat as X grows. Reject if distances do not decrease and stabilize around the theoretically expected limit across forms.
Novelty: It gives a concrete, testable distributional law for tensor products at the prime level with explicit SU(2)×SU(2) pushforward, not previously verified in finite-data regimes.

2) Finite-Size Convergence Rate C(k)/log X for Symmetric-Power Sato–Tate
Summary: For a non-CM GL(2) newform f, the Wasserstein-1 distance between empirical Satake angles of Sym^k(f) up to primes ≤ X and the theoretical Sym^k Sato–Tate law is asymptotically C(k)/log X with an explicit k-dependent constant.
For a smart layperson: The angles at each prime should follow a precise “random rotation” law after applying a k-fold symmetry operation. This predicts not just the limit, but the speed at which real data approach the limit as you look at more primes. The rate depends on how many folds (k) you apply.
Falsification: Compute Satake angles via Hecke eigenvalues for many non-CM newforms, push forward by Sym^k (using Chebyshev polynomials), and measure W1 distance to the theoretical law for X in a growing sequence. Perform log-log regression for distance vs log X; reject if the slope deviates significantly from -1 and if fitted constants are inconsistent across forms for fixed k.
Novelty: It proposes a precise, uniform finite-size convergence law (including the decay exponent and dependence on k) rather than only the classical limiting Sato–Tate statement.

3) Finite-Size 1-Level Density Correction for Rankin–Selberg L-functions
Summary: For L(s,f×g) with f,g non-CM GL(2) newforms, the 1-level density of low-lying zeros exhibits a finite-size correction of size A/ log Q(f×g) where A depends only on the symmetry type (orthogonal) and archimedean factors, not on arithmetic of f or g.
For a smart layperson: The spacings of the first few “notes” (zeros) of these L-functions should mimic those of a universal orchestra (random matrices), with a small, quantifiable, 1/log-size adjustment. The claim is that this small adjustment is universal across choices of f and g with the same broad symmetry.
Falsification: Numerically compute zeros near 1/2 for many f×g up to a conductor cutoff using mpmath/arb backends, aggregate 1-level densities for test functions, and fit the deviation from the RMT prediction vs log Q. Reject if the observed correction is not approximately linear in 1/log Q with a constant independent of f,g after conditioning on symmetry.
Novelty: It predicts a universal finite-conductor correction constant for Rankin–Selberg families, a refinement beyond existing asymptotic universality.

4) Entropy Gap Characterizing Cuspidality of Sym^2 Lifts
Summary: The prime-sign entropy of a_p(f) (viewed as signs of normalized coefficients) is strictly lower for GL(2) newforms with non-cuspidal Sym^2 lift (CM) than for those with cuspidal Sym^2, with a quantifiable gap stable across levels and weights.
For a smart layperson: If you binarize the prime data into plus/minus, CM forms show a more predictable pattern than non-CM ones. This predictability can be measured as lower information content (entropy). The entropy gap acts as a diagnostic for the deeper functorial nature of the symmetric-square lift.
Falsification: For CM and non-CM datasets (e.g., weight 2 newforms up to large level), compute normalized signs of a_p/2√p and estimate entropy rates via block or Lempel–Ziv estimators on primes p ≤ X. Reject if the estimated entropy gap diminishes to zero as X grows or is not consistently positive across families.
Novelty: It introduces an information-theoretic invariant of prime coefficients that cleanly separates cuspidal vs non-cuspidal symmetric-square functorial behavior.

5) Covariance Outlier Principle for Detecting Functorial Relations
Summary: The prime-wise covariance of normalized Hecke eigenvalues of two GL(2) newforms is asymptotically zero unless the forms are related by a twist or a known functorial relation; in related cases the covariance converges to an explicit nonzero constant.
For a smart layperson: Two unrelated musical scores (forms) have independent note patterns at primes; their up-and-down movements don’t line up overall. But if one secretly comes from the other by a known transformation, a persistent alignment shows up as nonzero covariance. This provides a numerical detector of hidden relationships.
Falsification: For a large set of pairs (f,g), compute empirical covariance of a_p(f)/(2√p) and a_p(g)/(2√p) over primes p ≤ X, and track convergence as X grows. Verify nonzero plateaus only for twisted or base-change-related pairs (confirmed via LMFDB metadata); reject if significant nonzero covariances appear generically or if known related pairs fail to show the predicted constant.
Novelty: It provides a quantitative, falsifiable covariance criterion that singles out functorial/twist relations from raw prime data.

6) Base-Change Detector via Split/Inert Coefficient Asymmetry
Summary: A GL(2) newform is a quadratic base-change from K if and only if the difference between average normalized coefficients over primes split vs inert in K converges to a nonzero constant determined by the base-change, otherwise the difference tends to 0.
For a smart layperson: If a form comes from a quadratic field, its prime fingerprints differ depending on how primes behave in that field—split or inert. Averaging the fingerprints separately exposes a persistent offset if and only if there’s a hidden origin in that field.
Falsification: For many candidate fields K and forms f, compute Δ_X(f,K) = mean_{p≤X, split}(a_p/2√p) − mean_{p≤X, inert}(a_p/2√p). Reject if Δ_X does not stabilize near a nonzero constant for known base-change forms or if it does not tend to 0 for forms known not to be base-change.
Novelty: It formulates a concrete split/inert asymmetry statistic with an if-and-only-if asymptotic, serving as a falsifiable numerical base-change characterization.

7) Optimal Transport Characterization of Functorial Pushforward
Summary: Among all measurable maps pushing the SU(2) Sato–Tate measure to the Sym^k Sato–Tate measure, the functorial Chebyshev pushforward uniquely minimizes the average squared transport cost on angle space.
For a smart layperson: There are many ways to distort one random “angle” law into another, but the lift prescribed by Langlands (the k-fold symmetry map) is the most economical on average, in a precise distance sense. It says functoriality solves an optimal matching problem between probability clouds.
Falsification: Sample many angles θ from SU(2) Sato–Tate via Python, compare the empirical transport cost under the Chebyshev map vs numerically optimized alternative maps (parameterized splines/polynomials) pushing forward the distribution. Reject if another map consistently yields lower average squared cost while preserving the correct target law.
Novelty: It reframes a local-global functorial pushforward as a unique optimizer of an optimal transport problem with a concrete, checkable cost criterion.

8) Cross-Moment Orthogonality Rate for Symmetric Powers
Summary: For non-CM f, the empirical covariance between normalized a_p(f) and normalized traces of Sym^m(f) over primes ≤ X decays as C_m/ log X with an explicit constant C_m=0 for m≠1 and a computable nonzero C_1, quantifying finite-size orthogonality.
For a smart layperson: Different “views” (symmetric powers) of the same signal should be uncorrelated in the limit, but at finite data sizes there is a small leftover correlation. This statement predicts exactly how small that correlation is and how fast it disappears as you include more primes.
Falsification: Compute sequences X_p= a_p(f)/(2√p) and Y_p^{(m)} = normalized trace of Sym^m at p via Chebyshev polynomials; for increasing X, estimate covariance and fit vs 1/log X. Reject if the decay is not linear in 1/log X or if the intercept is not statistically indistinguishable from 0 for m≠1 across forms.
Novelty: It gives a precise finite-size orthogonality law (with rate and constants) between base and lifted coefficient streams, not just asymptotic zero correlation.

9) Endoscopic Mixture Law for GSp(4) to GL(4) Transfers
Summary: For Siegel cusp forms whose automorphic representations are endoscopic, the distribution of their GL(4) lift prime coefficients is a fixed convex mixture of two explicit Sato–Tate measures determined by the endoscopic parameter.
For a smart layperson: Some higher-dimensional forms are built from simpler parts; their prime fingerprints should look like a weighted blend of two known random patterns. The weights are dictated by the internal symmetry data of the form.
Falsification: Using LMFDB data for GSp(4) eigenforms with known endoscopic status, compute normalized GL(4) lift coefficients at primes and fit the empirical distribution to a two-component mixture with fixed component densities; test goodness-of-fit and estimate weights. Reject if known endoscopic forms fail the mixture model or non-endoscopic forms spuriously fit it.
Novelty: It proposes a concrete, testable mixture-model signature of endoscopy at the prime-coefficient level for a specific functorial transfer.

10) Universal First-Zero Scaling for Rankin–Selberg Families
Summary: The expected height of the lowest nontrivial zero of L(s,f×g) above 1/2 scales as A/ log Q with A depending only on the symmetry type (orthogonal) and not on f,g beyond their conductors.
For a smart layperson: The first zero acts like a “ground note” whose pitch rises slowly with complexity; this predicts a simple 1/log rule with a universal coefficient for the whole family. It’s a sharp, testable refinement of the random matrix analogy.
Falsification: Numerically compute the first zeros for many L(s,f×g) across conductor bins, regress mean first-zero height against 1/ log Q, and check if slopes agree within error across bins and pairs. Reject if dependence on f or g beyond Q remains after controlling for symmetry type.
Novelty: It asserts and quantifies a universal finite-conductor scaling law for the first zero in Rankin–Selberg families, going beyond qualitative symmetry-type predictions.


        Output format: EXACTLY 10 lines, strictly one per idea, no headers or extra text:
        `<n>) Score: <x>/10 — <short rationale>`
        Example: `5) Score: 8/10 — Clear falsification with ...`
