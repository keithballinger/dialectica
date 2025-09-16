Selected idea #7:

7) MEV-Deterrence via Endogenous AMM Fees in a Repeated Game
Summary: Liquidity providers set adaptive AMM fees to deter toxic arbitrage (MEV), leading to cyclical fee equilibria and higher risk-adjusted LP returns.
For a smart layperson: In crypto markets, bots can exploit trades at others’ expense; liquidity providers can fight back by changing fees. This arms race settles into cycles where fees rise and fall to keep bots at bay, improving outcomes for providers overall.
Falsification: Build a Python market simulator with an AMM, LP agents choosing fees, and arbitrage bots exploiting price gaps; include stochastic order flow. Test for endogenous fee cycles and LP return improvements versus fixed-fee baselines; vary bot latency and gas costs. Absence of fee cycles or improved LP returns falsifies the claim.
Novelty: It applies entry-deterrence game theory to DeFi microstructure, yielding concrete, testable dynamics for fee-setting and MEV.
