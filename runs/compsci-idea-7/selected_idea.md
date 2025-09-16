Selected idea #7:

7) Percolation Threshold Governs In-Context Learning in Transformers
Summary: Strong in-context learning emerges when the per-layer attention graph exceeds a percolation threshold that guarantees low-stretch paths from prompt tokens to targets.
For a smart layperson: Transformers learn from the prompt by passing information along attention links; this theory says there’s a tipping point in how connected those links must be for the model to use examples effectively.
Falsification: Train families of sparse-attention Transformers with controlled sparsity patterns, measure per-layer percolation and path stretch, and evaluate in-context learning tasks; absence of threshold-like performance changes at predicted connectivity levels falsifies the theory.
Novelty: Explains in-context learning as a graph phase transition, replacing architecture-specific heuristics with a structural criterion.
