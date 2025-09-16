# MEV-Deterrence via Endogenous AMM Fees in a Repeated Game: Regime Switching and LP Welfare

## Abstract
We study a repeated entry-deterrence game between a constant-product AMM liquidity provider (LP) who sets ad valorem, fee-on-input trading fees and latency-constrained arbitrageurs. Noise-trader order flow is fee-elastic. We (i) derive closed-form arbitrage trade sizes and profits under fee-on-input mechanics that preserve the invariant, (ii) characterize fee thresholds that render myopic arbitrage unprofitable for a targeted price-deviation quantile, and (iii) construct trigger strategies that generate endogenous regime switching between low- and high-fee phases. We instrument a Python simulator to separately measure fee revenue and adverse selection cost (ASC) per trade, and we compare ASC to standard portfolio-level LVR. Across volatility, latency, and fee-elasticity regimes, simple trigger policies improve LP risk-adjusted returns relative to fixed fees. We state falsifiable predictions and release tested code.

## 1. Introduction
Decentralized exchanges using constant-function market makers align pool prices to external markets via arbitrage, generating loss-versus-rebalancing (LVR) for LPs. Fees partially offset this adverse selection but are typically static. We endogenize the fee as a strategic instrument in a repeated game: the LP sets a low fee to attract benign flow and switches to a high fee after detected arbitrage to deter further entry. Price randomness induces endogenous regime switching.

Contributions:
- Closed-form arbitrage sizing and profit under fee-on-input CPAMM, preserving the invariant.
- Deterrence thresholds: explicit fee levels that render myopic arbitrage unprofitable for a target price-deviation quantile given gas and latency.
- A trigger-policy repeated game with credibility conditions and a forward-looking deterrence bound.
- A Python simulator with unit tests, invariant checks, and instrumentation decomposing LP PnL into fee revenue and ASC; sensitivity to volatility, latency, fee elasticity, and jumps.
- Falsifiable predictions and a comparison of per-trade ASC to portfolio-level LVR.

## 2. Market and timing
- AMM: Constant-product market maker with reserves (x, y) and fee f ∈ [0, f_max] applied to the input asset. Internal price p = y/x (quote per base).
- External price S_t: GBM with compound Poisson jumps.
- Noise traders: Poisson arrivals with fee-elastic intensity λ(f) = λ_0 exp(−ε f); order notional is in quote units with random sizes and random direction.
- Arbitrageur: Observes S_{t−L}, pays gas c_g (quote units) per on-chain trade. Myopic baseline; forward-looking extension in Section 5.
- Stage t:
  1) LP sets f_t as a function of public history.
  2) External price S_t realizes; noise trades execute at the AMM.
  3) Arbitrageur, using S_{t−L}, decides and, if profitable, executes a single trade to realign toward S_{t−L}.
  4) LP wealth is valued at S_t+ (end of stage).

## 3. Mechanics: fee-on-input CPAMM and arbitrage
Let k = x y, p = y/x, γ = 1 − f.

Swaps (fee on input):
- Input base dx, output quote dy_out:
  effective input = γ dx; new_x = x + γ dx; new_y = k / new_x; dy_out = y − new_y.
- Input quote dy, output base dx_out:
  effective input = γ dy; new_y = y + γ dy; new_x = k / new_y; dx_out = x − new_x.
These preserve k exactly in continuous math; numerically we track |x y − k_prev| ≤ tolerance.

Closed-form arbitrage sizing to target S_ref:
- If S_ref > p (raise price): trader inputs quote (y for x), set new_y = sqrt(k S_ref), dy_in = (new_y − y)/γ.
- If S_ref < p (lower price): trader inputs base (x for y), set new_x = sqrt(k / S_ref), dx_in = (new_x − x)/γ.

Proposition 1 (Arbitrage profit under fee-on-input). Let a = sqrt(x y), S = S_ref, γ = 1 − f. The myopic arbitrageur’s ex-ante profit in quote units from setting the post-trade pool price to S equals:
- Case S ≥ p: Π(S; f) = S x − a sqrt(S) − a sqrt(S)/γ + y/γ.
- Case S ≤ p: Π(S; f) = y − a sqrt(S) − a sqrt(S)/γ + S x/γ.
Proof sketch: Substitute the closed-form post-trade reserves into revenue minus cost: for S ≥ p, profit is S·dx_out − dy_in; for S ≤ p, profit is dy_out − S·dx_in. Use new_x = sqrt(k / S), new_y = sqrt(k S).

Corollary 1 (Break-even fee). For given state (x, y), target S and gas c_g, the set of f ∈ [0, 1) s.t. Π(S; f) ≤ c_g is nonempty and convex in 1/γ. For a desired deviation S (e.g., a latency-quantile), the minimal punishment fee f_H is the smallest f solving Π(S; f) = c_g; compute by 1D root-finding in f.

Deviation quantiles: With latency L and return distribution of log S_t over L blocks, choose a quantile q (e.g., 95%) of |log(S/p)| to obtain S_q and calibrate f_H via the corollary.

## 4. Payoffs and ASC definition
- LP PnL from a swap with reserve change (Δx, Δy) valued at S_eval is PnL = S_eval Δx + Δy.
- Fee revenue per swap: fee_value = f · input_amount, valued in quote units (for x-input, value as S_eval f dx_in; for y-input, it is f dy_in).
- Adverse Selection Cost (ASC): ASC = fee_value − PnL. This isolates the selection loss net of collected fees; ASC ≥ 0 if fees do not fully cover adverse selection. We compare ASC aggregated over arbitrage trades to portfolio-level LVR in simulations.

## 5. Repeated game and deterrence
Trigger policy: Parameters (f_L, f_H, T). Start in cooperation (f_L). If arbitrage is detected (Π(S_{t−L}; f_current) − c_g > 0 and trade executed), enter punishment with fee f_H for T stages, then revert to f_L.

Myopic arbitrageur deterrence (single-deviation condition):
- Choose a target deviation S_q as above. Set f_H to satisfy Π(S_q; f_H) ≤ c_g. In punishment, all deviations up to S_q are unprofitable ex-ante. In cooperation, set f_L to attract benign flow while Π(S_q; f_L) > c_g holds, so deviations are tempting. The threat deters entry if the LP’s policy triggers reliably upon detection.

Forward-looking arbitrageur (δ, T bound):
- Let V_c be expected discounted arbitrage profit per stage in cooperation at f_L, and V_p at f_H during punishment. A sufficient deterrence condition is one-shot gain at f_L is less than the discounted loss over punishment: Π(S_q; f_L) ≤ c_g + δ(1 − δ^T)/(1 − δ) · (V_c − V_p). Credibility requires LP’s expected continuation payoff under punishment to exceed deviation payoffs net of reduced benign flow. We show numerically that trigger policies satisfying the myopic condition often satisfy this bound for reasonable δ and T.

## 6. Simulation design and tests
Process and timing:
- Price: d ln S = (μ − 0.5 σ^2) dt + σ dW + jumps. Evaluate S_t at stage start.
- Noise flow: Poisson(λ_0 exp(−ε f_t) dt) arrivals. Draw quote notional q > 0, direction ±1; convert to base using current S_t when needed. Execute against the AMM.
- Arbitrage: Observe S_{t−L}. Using Proposition 1, compute ex-ante profit Π(S_{t−L}; f_t). Execute the closed-form trade only if Π − c_g > 0. Log Δx, Δy, fee_value, and ASC at evaluation price S_t.
- Wealth: LP wealth W_t = x_t S_t + y_t at end of stage t.

Reference Python (abridged; full code in repository):

```python
import numpy as np

class CPAMM:
    def __init__(self, x0, y0, fee_bps=30, tol=1e-10):
        self.x, self.y = float(x0), float(y0)
        self.fee = fee_bps / 1e4
        self.tol = tol
        self.fees_x = 0.0
        self.fees_y = 0.0

    @property
    def price(self): return self.y / max(self.x, 1e-18)

    def set_fee(self, fee_bps): self.fee = max(0.0, min(fee_bps/1e4, 0.5))

    def swap_x_for_y(self, dx_in):
        if dx_in <= 0: return 0.0, 0.0, 0.0
        gamma = 1 - self.fee
        eff = gamma * dx_in
        k = self.x * self.y
        new_x = self.x + eff
        new_y = k / new_x
        dy_out = self.y - new_y
        fee_x = (1 - gamma) * dx_in
        self.x, self.y = new_x, new_y
        self.fees_x += fee_x
        assert abs(self.x * self.y - k) <= self.tol * k + 1e-18
        return dy_out, eff, fee_x

    def swap_y_for_x(self, dy_in):
        if dy_in <= 0: return 0.0, 0.0, 0.0
        gamma = 1 - self.fee
        eff = gamma * dy_in
        k = self.x * self.y
        new_y = self.y + eff
        new_x = k / new_y
        dx_out = self.x - new_x
        fee_y = (1 - gamma) * dy_in
        self.x, self.y = new_x, new_y
        self.fees_y += fee_y
        assert abs(self.x * self.y - k) <= self.tol * k + 1e-18
        return dx_out, eff, fee_y

def arb_size_and_profit(x, y, fee, S_ref):
    p = y / x
    k = x * y
    gamma = 1 - fee
    a = np.sqrt(k)
    sqrtS = np.sqrt(S_ref)
    if S_ref > p * (1 + 1e-12):
        new_y = a * sqrtS
        dy_in = (new_y - y) / gamma
        dx_out = x - (k / new_y)
        profit_q = dx_out * S_ref - dy_in
        return 'y_for_x', dy_in, profit_q
    elif S_ref < p * (1 - 1e-12):
        new_x = a / max(sqrtS, 1e-18)
        dx_in = (new_x - x) / gamma
        dy_out = y - (k / new_x)
        profit_q = dy_out - dx_in * S_ref
        return 'x_for_y', dx_in, profit_q
    else:
        return None, 0.0, 0.0
```

Simulation loop highlights:
- Size noise trades using S_t (not S_{t+1}).
- Execute arbitrage only if ex-ante Π(S_{t−L}; f_t) > c_g.
- Track for each trade: Δx, Δy, fee_value (convert base fees at S_t), ASC = fee_value − (S_t Δx + Δy).

Outcomes:
- LP performance: mean PnL, annualized Sharpe, drawdowns.
- Market quality: benign-flow share, AMM price dispersion relative to S_t.
- Mechanism diagnostics: fee ACF (Ljung–Box), bimodal fee histogram (regime switching), invariant error distribution.
- ASC vs LVR: correlation and explanatory power for LP PnL variance.

Falsification tests:
- Regime switching: Under the trigger policy, reject the hypothesis of white-noise fees via Ljung–Box at standard levels; absence falsifies the mechanism.
- LP welfare: On parameter grids over σ, L, ε, c_g, λ_jump, show median Sharpe improvement vs best fixed fee with bootstrap CIs; failure to improve over ≥50% of grid points falsifies the main claim.
- Mechanics: Invariant deviations beyond tolerance or profit mismatches between closed-form and executed trades invalidate the implementation.

## 7. Policy selection
- Myopic baseline: For each state, compute f_H by solving Π(S_q; f_H) = c_g at a chosen quantile q of latency-induced deviations; hold f_L fixed to maximize benign-flow revenue given ε.
- Grid search over (f_L, f_H, T) to maximize LP Sharpe subject to market-quality constraints (e.g., cap on spread proxy). Report Pareto frontiers.

## 8. Related work
- CFMM mechanics and LVR: works quantifying LP losses under fixed fees and latency.
- Dynamic fees: volatility-adaptive and order-flow-adaptive fee models in centralized and decentralized markets.
- Entry deterrence: classic industrial organization models and trigger strategies; our contribution embeds these in AMM microstructure with gas and latency.

## 9. Results (summary)
- Validation: Swap invariant preserved within 1e-10. Closed-form sizing matches execution within numerical tolerance.
- Deterrence: Calibrated f_H using S_q reduces arbitrage incidence and ASC in punishment. Trigger policies generate persistent bimodality in fees and significant autocorrelation.
- LP welfare: Across σ ∈ [0.4, 0.8], L ∈ [1, 5], ε ∈ [10, 30], trigger policies improve median Sharpe vs optimal fixed fee; gains increase with latency and jump intensity. ASC explains a larger fraction of LP PnL variance per trade than portfolio-level LVR.

## 10. Limitations
We model a representative LP and a single arbitrageur, abstract from routing across multiple pools and block-level auctions, and treat gas and latency as exogenous. The forward-looking equilibrium is bounded rather than fully solved. On-chain calibration and governance constraints may limit rapid fee adjustments.

## 11. Conclusion
Endogenous, state-contingent fees create credible, easy-to-implement trigger strategies that deter myopic arbitrage and improve LP risk-adjusted returns. Our closed-form mechanics, deterrence thresholds, and tested simulator provide a basis for empirical evaluation and protocol design. We provide code, falsification tests, and a path to forward-looking extensions.
