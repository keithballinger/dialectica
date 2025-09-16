Selected idea #2:

2) The Exact Privacy–Capacity Frontier under (ε,δ)-Differential Privacy
Summary: The maximum mutual information between a dataset and a privatized output under (ε,δ)-differential privacy equals a closed-form function g(ε,δ) independent of the data distribution and is achieved by staircase mechanisms.
For a smart layperson: Differential privacy limits how much information about any individual leaks. This theory gives the exact best-possible information transfer under a given privacy level, not just a bound. It also identifies a simple family of mechanisms that achieve this limit.
Falsification: Construct any (ε,δ)-DP mechanism and empirically/analytically estimate mutual information for worst-case priors; if MI exceeds g(ε,δ) for any ε,δ, the theory is false, or if staircase mechanisms fail to approach g(ε,δ), it is refuted.
Novelty: It claims an exact, prior-free privacy–information curve and its achievers, surpassing existing upper/lower bounds.
