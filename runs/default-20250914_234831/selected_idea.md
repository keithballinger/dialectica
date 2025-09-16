Selected idea #10:

10) Directionality-Asymmetric CX Errors Follow Spectator-Activation Law
Summary: CX error asymmetry between control→target directions grows with the number of actively driven spectator qubits within two edges, due to cross-drive and residual ZZ.
For a smart layperson: A two-qubit gate can be worse in one direction, and it gets even worse if neighbors are busy at the same time. This proposes a simple rule: the more nearby activity, the bigger the asymmetry.
Falsification: For a fixed pair, perform interleaved RB of CX in both directions while activating 0, 1, 2, … nearby spectators with random single-qubit Clifford layers; plot EPC asymmetry vs number activated. A positive, roughly linear trend supports; flat trend falsifies.
IBM cost plan: 4 spectator-load settings × 2 directions × 24 sequences × 1000 shots (~192k shots); under $100 on mid-scale backends.
Novelty: Predicts a quantitative, connectivity-dependent law for CX directionality under realistic concurrent activity.
