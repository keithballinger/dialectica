1) Algorithmic Transparency Thresholds Create Tacit Collusion in Price Games
Summary: There exists a critical joint threshold in market transparency and algorithm learning speed beyond which independent pricing bots converge to supra-competitive prices without communication.
For a smart layperson: As firms adopt pricing algorithms, seeing competitors’ prices and learning quickly can accidentally make them keep prices high. The idea predicts a tipping point where transparency plus fast learning pushes markets from competition to tacit collusion. Below that point, prices remain competitive.
Falsification: Implement Python simulations of repeated Bertrand competition with Q-learning/Deep-Q agents, varying transparency (signal noise on rival prices) and learning rate. Measure average price relative to Nash and identify a phase transition (sharp change) and scaling in variance/autocorrelation near the threshold. Attempt to break the effect by adding exploration or price caps; failure to do so refutes robustness.
Novelty: It predicts and characterizes a universal collusion threshold tied jointly to observability and learning dynamics, not previously formalized with phase-transition diagnostics.

2) Liquidity Centrality Governs Shock Pass-Through in Production Networks
Summary: A firm’s pass-through of cost shocks equals a monotone function of its liquidity-adjusted network centrality, altering equilibrium markups and shock propagation.
For a smart layperson: Firms buy from and sell to each other in networks, and some have more cash buffers than others. The claim is that a firm’s position in the network and its cash on hand together determine how much it raises prices when costs go up. This gives a simple rule to predict which firms amplify inflationary shocks.
Falsification: Build a Python agent-based input–output network with bilateral bargaining and working-capital constraints; shock a subset of suppliers. Compute pass-through by node and compare to predictions from a closed-form liquidity-centrality index; vary liquidity distributions and network structure to test invariance. Reject if pass-through does not increase with the proposed index.
Novelty: It fuses bargaining-theoretic pricing with financial frictions to yield a new, testable centrality metric for pass-through.

3) Menu Complexity Breaks Revenue Equivalence via Attention Constraints
Summary: Above a menu complexity threshold, bounded-attention bidders induce systematic revenue shortfalls versus standard auction theory, selecting new equilibria.
For a smart layperson: When auctions offer too many pricing options or variants, real bidders can’t process them all and switch to shortcuts. This predicts a tipping point where seller revenue drops below textbook expectations. Simpler menus avoid this revenue cliff.
Falsification: Simulate auctions in Python with bidders modeled as limited-complexity agents (e.g., k-sparse strategy space or limited-lookahead) across increasing menu sizes and dimensionality. Estimate the revenue curve and detect the threshold where it deviates from Myerson’s benchmark; perturb attention capacity to test the mechanism. Failure to find a threshold or sensitivity to irrelevant changes refutes the theory.
Novelty: It endogenizes attention into auction games, producing a falsifiable complexity threshold for revenue equivalence failure.

4) Noisy Ratings Induce Wage Compression in Dynamic Matching Markets
Summary: When public ratings of workers are sufficiently noisy, repeated matching equilibria collapse to pooling wages, reducing sorting and welfare.
For a smart layperson: If online ratings are too unreliable, employers can’t tell good from bad workers and end up paying everyone similar wages. That hurts both top workers and overall efficiency. Cleaner signals restore differences and better matches.
Falsification: Code a Python simulation of a two-sided matching market with repeated gigs, Bayesian employers, and noisy public ratings; vary noise and persistence. Measure wage dispersion, match quality, and surplus; identify a critical noise level where dispersion collapses. Attempt to rescue dispersion with private signals; if dispersion remains, the mechanism is wrong.
Novelty: It links cheap-talk/washer-noise signaling games to dynamic matching, predicting a sharp noise threshold for wage compression.

5) Verification Cycles in Carbon Offsets from Strategic Signal Investment
Summary: Endogenous verification investments by offset buyers and sellers generate predictable cycles in offset prices and realized abatement quality.
For a smart layperson: In carbon markets, both sides can spend money to prove projects are real, but they time and adjust these efforts strategically. That dance creates boom–bust cycles in prices and the actual environmental benefit delivered. The model explains when and why those cycles occur.
Falsification: Simulate in Python an offset market with heterogeneous project qualities, verification lags, and strategic budget allocation to verification vs. production. Test for cyclical price and quality dynamics at frequencies tied to verification lag and cost; remove strategic choice to see cycles disappear. If cycles don’t emerge or don’t align with lag structure, reject.
Novelty: It brings dynamic signaling investment into permit markets, generating falsifiable spectral predictions for prices and quality.

6) Stackelberg Surge Pricing Stabilizes Congestion but Creates Endogenous Price Oscillations
Summary: Public surge-pricing rules act as a commitment device that lowers average wait times while inducing persistent oscillatory equilibria in ride-hailing markets.
For a smart layperson: When a platform announces how prices surge with demand, drivers and riders react to that rule. This can cut average waits but also produce price waves as people overreact in cycles. Oscillations are thus an expected side effect, not a malfunction.
Falsification: Implement a Python simulation with rider demand, driver supply entry/exit frictions, and a published surge function; compare to fixed pricing. Measure welfare, wait times, and spectral density of prices; vary adjustment speeds to map oscillation amplitude. If no oscillations appear under plausible parameters, the commitment mechanism is falsified.
Novelty: It integrates congestion games and dynamic mechanism design to predict oscillations as an equilibrium property of public pricing rules.

7) MEV-Deterrence via Endogenous AMM Fees in a Repeated Game
Summary: Liquidity providers set adaptive AMM fees to deter toxic arbitrage (MEV), leading to cyclical fee equilibria and higher risk-adjusted LP returns.
For a smart layperson: In crypto markets, bots can exploit trades at others’ expense; liquidity providers can fight back by changing fees. This arms race settles into cycles where fees rise and fall to keep bots at bay, improving outcomes for providers overall.
Falsification: Build a Python market simulator with an AMM, LP agents choosing fees, and arbitrage bots exploiting price gaps; include stochastic order flow. Test for endogenous fee cycles and LP return improvements versus fixed-fee baselines; vary bot latency and gas costs. Absence of fee cycles or improved LP returns falsifies the claim.
Novelty: It applies entry-deterrence game theory to DeFi microstructure, yielding concrete, testable dynamics for fee-setting and MEV.

8) Strategic Obfuscation as Commitment in Data Markets
Summary: Data sellers optimally degrade preview access to shift buyer beliefs and raise prices, producing mixed-strategy obfuscation equilibria and welfare losses.
For a smart layperson: Sellers of datasets sometimes hide details before sale; this can be deliberate to make the data look more valuable. The theory says there’s a stable mix between showing and hiding that keeps prices up but hurts overall efficiency.
Falsification: Simulate buyers performing Bayesian learning with outside options and sellers choosing obfuscation rates affecting signal precision; implement repeated posted-price or auction settings in Python. Estimate equilibrium obfuscation and prices across costs; verify mixed strategies and welfare losses. If prices don’t respond to obfuscation or no mixing occurs, reject.
Novelty: It unifies information design and IO of data as a commodity, delivering a precise, testable obfuscation–price linkage.

9) Capacity as Punishment Device Generates Bistable Price Dynamics
Summary: In repeated price competition with depreciating capacity investments, equilibria switch between competitive and collusive regimes depending on the depreciation rate, exhibiting hysteresis.
For a smart layperson: Firms can invest in capacity that lets them punish price cuts by flooding the market later. If capacity decays slowly, high-price cooperation can stick; if it decays fast, competition dominates. History matters: once in one regime, it’s hard to switch.
Falsification: Write a Python dynamic game with investment, depreciation, and pricing; simulate equilibrium via reinforcement learning or value iteration. Vary depreciation and adjustment costs; detect bistability and hysteresis loops in price–capacity space. Failure to find regime switching or path dependence refutes the theory.
Novelty: It ties dynamic capital accumulation to trigger strategies to predict testable hysteresis in oligopoly pricing.

10) Networked P2P Insurance Sustains Only Above a Reciprocity–Audit Threshold
Summary: Peer-to-peer insurance on social networks exhibits a sharp sustainability threshold in reciprocity and audit cost, below which strategic default unravels the market.
For a smart layperson: In community insurance, people can be tempted to cheat on claims unless peers can cheaply check and value long-term relationships. The model predicts a tipping point in trust and monitoring needed to keep premiums low and the system alive.
Falsification: Simulate a Python network of agents facing stochastic losses, premium pooling, strategic default, and social audits with costs; vary reciprocity (future discounting) and audit cost. Identify the threshold where default spikes and premiums blow up; test robustness to network topology. Absence of a sharp threshold falsifies the claim.
Novelty: It combines repeated games with network economics to generate a clear, testable sustainability condition for P2P insurance.
