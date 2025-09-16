Major Revisions

Title
An Entropy Gap Diagnostic for Cuspidality of Symmetric-Square Lifts

Abstract
We propose an information-theoretic invariant of prime Hecke data that distinguishes GL(2) newforms with non-cuspidal symmetric-square lifts (i.e., CM forms) from those with cuspidal symmetric-square lifts (non-CM). For a newform f, we encode S(p) ∈ {−, 0, +} as the sign of the normalized Hecke eigenvalue a_p(f)/(2 p^{(k−1)/2}). We define the minimum conditional sign entropy E_min(f; X) as the empirical conditional entropy Ĥ(S | Q_Δ) over primes p ≤ X, minimized over quadratic characters χ_Δ (via their inert/split indicator Q_Δ(p) ∈ {0,1}). Under Sato–Tate and Chebotarev assumptions, we predict an asymptotic entropy gap: E_min(f; X) → 1/2 bit for CM forms (reflecting deterministically zero coefficients at inert primes), whereas E_min(f; X) → 1 bit for non-CM forms (reflecting conditional independence of signs from any quadratic splitting mask). We develop estimators based on block entropies and Lempel–Ziv (LZ76) complexity and provide a falsification-oriented Python workflow using Sage/LMFDB to compute a_p(f) and estimate the gap across weights and levels. This “masked sign entropy” connects functorial structure of Sym^2 f (isobaric constituent for CM) to a quantitative, data-driven invariant stable across families.

Introduction
- Background. For a holomorphic newform f on GL(2)/Q of weight k ≥ 2 and level N, Gelbart–Jacquet established functoriality of Sym^2 f to an automorphic representation on GL(3). The lift is cuspidal unless f has complex multiplication (CM), in which case Sym^2 f is Eisenstein (isobaric sum containing a character). At unramified primes p, a_p(f) encodes local Frobenius data. For CM forms associated to a Hecke character of an imaginary quadratic field K, a_p(f) = 0 for inert p (density 1/2); for non-CM forms, Sato–Tate implies normalized a_p(f) have continuous distribution with no atom at 0.
- Motivation. Despite deep theoretical criteria for cuspidality of Sym^2 f, practical data-driven diagnostics are valuable for large-scale computations and as empirical probes of functorial structure. Pure sign-only statistics are robust to scaling and can be computed at large X. However, naive single-letter Shannon entropy H(sign(a_p)) does not cleanly separate CM vs non-CM when zeros are ignored or removed. We show that conditioning on a quadratic splitting mask—searching over χ_Δ—recovers a robust entropy gap reflecting the non-cuspidal structure.
- Contribution. We introduce the minimum conditional prime-sign entropy E_min(f; X) and predict a universal gap of roughly 1/2 bit between CM and non-CM forms. We provide:
  1) A principled definition tied to the quadratic subfield underlying CM;
  2) A heuristic derivation of the limiting values under standard equidistribution hypotheses;
  3) Finite-sample estimators (block/Shannon, LZ76) and a falsification-driven experimental protocol implemented in Python/Sage using LMFDB.

Method
Setup and normalization
- For a newform f of weight k and level N, define for primes p ∤ N:
  S_f(p) = sign(a_p(f)/(2 p^{(k−1)/2})) ∈ {−1, 0, +1}, with 0 if a_p(f) = 0.
- For a fundamental discriminant Δ, let χ_Δ be the real quadratic character and define the inertness indicator Q_Δ(p) = (1 − χ_Δ(p))/2 ∈ {0,1}; here Q_Δ(p)=1 for inert primes in K = Q(√Δ) and Q_Δ(p)=0 for split primes (ramified primes are finite and can be discarded).

Entropy functionals
- Empirical single-letter conditional entropy given Δ:
  Ĥ1(S | Q_Δ; X) = − Σ_{q∈{0,1}} Σ_{s∈{−,0,+}} P̂_X(Q_Δ=q, S=s) log2 P̂_X(S=s | Q_Δ=q).
- Minimum conditional entropy:
  E_min(f; X; B) = min_{|Δ| ≤ B, Δ fundamental} Ĥ1(S | Q_Δ; X).
- Block refinements: For k ≥ 1, define S-blocks over consecutive primes B_i = (S(p_i),…,S(p_{i+k−1})), and similarly Q-blocks; estimate Ĥk(S | Q_Δ) using empirical k-block frequencies. The entropy rate is limsup_{k→∞} Ĥk/k. In practice, we use k ≤ 8.
- Algorithmic complexity proxy: Lempel–Ziv (LZ76) complexity C_LZ of the trinary sequence S(p) after mapping {−,0,+} to a 3-letter alphabet; conditional LZ is estimated by compressing S jointly with Q_Δ via interleaving or two-stream compression and recording the additional bits per symbol.

Heuristic prediction (Entropy Gap Principle)
- CM case. If f has CM by K with discriminant Δ_K, then S_f(p) = 0 exactly on the inert primes, i.e., when Q_ΔK(p)=1; on split primes Q_ΔK(p)=0, the signs of a_p are approximately symmetric ± with no atom at 0. Hence asymptotically:
  H1(S | Q_ΔK) ≈ P(Q=1)·H(S|Q=1) + P(Q=0)·H(S|Q=0)
                 ≈ (1/2)·0 + (1/2)·1 = 1/2 bit,
  where we used density 1/2 for inert primes and H(±)=1 bit on split primes.
  Block/language-level refinements do not increase this limit because zeros are deterministically predicted by Q_ΔK.
- Non-CM case. Under Sato–Tate and standard independence heuristics, for any fixed quadratic character χ_Δ the sign distribution is asymptotically ± with probability 1/2 independent of Q_Δ, and zeros have density 0. Thus H1(S | Q_Δ) → 1 bit for all Δ, so E_min(f; X; B) → 1 bit as X→∞, B→∞.
- Gap. Therefore E_min tends to 1/2 bit for CM, to 1 bit for non-CM, producing an asymptotic gap of 1/2 bit. This is a manifestation of functoriality: non-cuspidality of Sym^2 f in the CM case corresponds to the presence of a one-dimensional summand whose quadratic field controls inert primes exactly, yielding a low-entropy mask predicting zeros.

Remarks
- Why conditional entropy rather than unconditional? Unconditional tri-symbol entropy H1(S) is larger for CM (because of the third symbol 0 with density 1/2). Conditioning removes the trivial symbol-inflation and exposes predictability arising from the quadratic summand.
- Independence assumptions. Rigorous proofs of entropy rates would require quantitative Sato–Tate and independence across prime splitting conditions. Our results are framed as a falsifiable empirical law motivated by proven equidistribution and the known structure of CM.

Experiments (falsification plan)
Data acquisition
- Forms:
  - Weight k ∈ {2, 4, 6, 8, 12}, levels N up to 4000, both CM and non-CM (CM detection via LMFDB tags or by checking vanishing density of a_p).
  - Exclude primes p | N.
- Coefficients:
  - For each f, compute a_p(f) for primes p ≤ X (X from 10^4 up to ≥ 10^6 when feasible).
  - Normalize a_p by 2 p^{(k−1)/2}.
- Quadratic masks:
  - Enumerate fundamental discriminants Δ with |Δ| ≤ B(X) (e.g., B = X^0.6 or fixed caps such as 10^5, 10^6).
  - For each Δ, compute Q_Δ(p) via Kronecker symbol (Δ/p).

Estimators
- Ĥ1(S | Q_Δ; X) from counts; E_min is the minimum over Δ.
- Ĥk for k=2,…,8 using sliding windows; report Ĥk/k.
- LZ76 complexity per prime for S, and conditional variant by compressing the pair (Q_Δ, S) (e.g., interleaving symbols) and subtracting the standalone complexity of Q_Δ.

Statistical safeguards
- Delta-method standard errors for Ĥ1 from multinomial counts.
- Stability checks:
  - Vary X along a doubling schedule; plot E_min vs log X.
  - Vary B; check that E_min plateaus once Δ_K enters the scan for CM forms.
  - Cross-validation: split primes into disjoint ranges to avoid overfitting Δ.
- Falsification criteria:
  - Reject if, across forms, the CM vs non-CM median E_min gap shrinks to 0 as X grows.
  - Reject if a substantial fraction of non-CM forms exhibit E_min < 0.8 bits persistently as X increases and B grows.
  - Reject if CM forms’ E_min does not stabilize near 0.5 ± 0.05 bits once Δ_K is scanned.

Python/Sage workflow (reproducible skeleton)
- Sage setup for coefficients:
  from sageall import Newforms, Gamma0, prime_range, kronecker_symbol
  import math, itertools, statistics
  def normalized_signs(f, X):
      # f is a Sage newform unique up to Galois; pick a rational coefficient embedding if needed
      # get q-expansion up to X
      qs = f.q_expansion(X+10)  # enough terms
      primes = list(prime_range(2, X+1))
      S = []
      for p in primes:
          ap = qs[p]
          # handle embeddings: take real part or specify rational newforms
          ap = complex(ap).real
          if ap == 0:
              S.append(0)
          else:
              norm = 2*(p**((f.weight()-1)/2))
              v = ap/norm
              S.append(1 if v > 0 else -1)
      return primes, S

  def H1_cond(S, Q):
      # S in {-1,0,1}, Q in {0,1}
      import math
      from collections import Counter
      n = len(S)
      cnt = {(q,s):0 for q in (0,1) for s in (-1,0,1)}
      cQ = {0:0,1:0}
      for s,q in zip(S,Q):
          cnt[(q,s)] += 1
          cQ[q] += 1
      def H(probs):
          return -sum(p*math.log2(p) for p in probs if p>0)
      h = 0.0
      for q in (0,1):
          if cQ[q]==0: continue
          probs = [cnt[(q,s)]/cQ[q] for s in (-1,0,1)]
          h += (cQ[q]/n)*H(probs)
      return h

  def entropy_min_over_discriminants(primes, S, B):
      # scan fundamental discriminants Δ up to |Δ|≤B
      def fundamental_discriminants(B):
          Ds = []
          for D in range(-B, B+1):
              if D==0: continue
              if kronecker_symbol(D,1)==1 and D == fundamental_discriminant(D):
                  Ds.append(D)
          return Ds
      Ds = []
      # Sage has fundamental_discriminants in quadratic_forms; otherwise supply a list externally.
      # For brevity, assume we have a list Ds of fundamental discriminants |Δ|≤B.
      best = (None, 10.0)
      for D in Ds:
          Q = [ (1 - kronecker_symbol(D,p))//2 for p in primes ]
          h = H1_cond(S, Q)
          if h < best[1]:
              best = (D, h)
      return best

  # LZ76 complexity for a finite alphabet:
  def lz76_complexity(seq):
      # seq is a list of small ints; returns c(n)/n
      s = ''.join({-1:'A', 0:'B', 1:'C'}[x] for x in seq)
      i, k, l, n = 0, 1, 1, len(s)
      c = 1 if n>0 else 0
      while True:
          if i + k == n or s[i+k] != s[l+k]:
              if k > l:
                  l += 1
              else:
                  c += 1
                  i += k
                  if i == n: break
                  l, k = 1, 1
          else:
              k += 1
              if l + k > n:
                  c += 1
                  break
      return c / max(1, n)

- LMFDB-backed alternative (for environments without Sage):
  - Query coefficients via the LMFDB API for specific newforms (weight, level, CM flag), or for elliptic curves (weight 2) using a_p = p + 1 − #E(F_p).
  - Use sympy for primes and kronecker_symbol.

Evaluation protocol
- For each form:
  - Compute E_min(f; X; B) at X ∈ {10^4, 3×10^4, 10^5, 3×10^5, 10^6}, with B ∈ {10^3, 10^4, 10^5, 10^6}.
  - Record the Δ achieving the minimum; for CM forms this should match Δ_K.
  - Compute block-normalized entropies Ĥk/k and LZ complexity; compare CM vs non-CM medians.
- Aggregate:
  - Plot E_min vs log X for CM vs non-CM cohorts; expect separation near 0.5 vs 1 bits.
  - Report misclassification rates using a decision rule: “predict CM if E_min ≤ 0.75 bits at X ≥ 10^5”.

Discussion
- Functorial signal in entropy. The non-cuspidality of Sym^2 f for CM forms manifests as a one-dimensional constituent whose quadratic field governs local vanishing of a_p at inert primes. The conditional entropy functional isolates this structure by revealing a low-entropy mask (Q_Δ) that predicts the zero symbol perfectly and leaves only a fair ± process on split primes. Non-CM forms lack such a global mask; conditioning on any quadratic character does not reduce sign entropy.
- Stability across weight and level. The gap is predicted to be weight- and level-stable because it depends only on the presence (or absence) of the quadratic constituent of Sym^2 f, not on the finer nebentype or conductor. Finite-sample fluctuations decrease as X grows.
- Relationship to Sato–Tate and Chebotarev. The 1/2–1 bit dichotomy follows from: (i) density 1/2 of inert primes in K (Chebotarev), (ii) symmetric sign distribution on split primes and absence of a 0-atom for non-CM forms (Sato–Tate), and (iii) approximate independence of signs from quadratic splitting for non-CM forms.

Limitations
- Heuristic independence. We use Sato–Tate and heuristic independence between signs and splitting symbols. Proving entropy rate limits would require finer exponential sum bounds and effective Sato–Tate along Chebotarev subsets.
- Finite computation. At moderate X, scanning Δ up to B may miss Δ_K for large conductors; in practice, CM conductors are small, but a principled rule for B(X) is needed. Cross-validation mitigates overfitting Δ on finite data.
- Embedding issues. For non-rational coefficient fields, choosing an embedding to compare real parts is standard but introduces negligible ambiguity for the sign after normalization; nonetheless, care is needed.
- Bad primes. We exclude p | N; including them complicates local normalization and does not impact the asymptotic entropy gap.

Conclusion
We introduce the minimum conditional prime-sign entropy as a data-driven invariant that detects the cuspidality of the symmetric-square lift of a GL(2) newform. The functorial difference between CM and non-CM forms yields an explicit, stable entropy gap: asymptotically 1/2 bit for CM (non-cuspidal Sym^2) versus 1 bit for non-CM (cuspidal Sym^2). We provide a falsification-oriented computational framework in Python/Sage to test this prediction across weights and levels using block entropy and LZ complexity estimators. Beyond offering a practical diagnostic, the result illustrates how functorial structure leaves quantitative information-theoretic signatures in prime coefficient data.

References (indicative)
- Gelbart, S.; Jacquet, H. A relation between automorphic representations of GL(2) and GL(3), Ann. Sci. ÉNS (1978).
- Serre, J.-P. A course in arithmetic. (Chebotarev background).
- Barnet-Lamb, T.; Geraghty, D.; Harris, M.; Taylor, R. A family of Calabi–Yau varieties and potential automorphy II, Publ. Math. IHÉS (2011). (Sato–Tate for non-CM).
- Murty, M. Ram. Oscillations of Fourier coefficients of modular forms. (Sign changes).
