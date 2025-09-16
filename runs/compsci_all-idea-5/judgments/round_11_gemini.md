Publish.

The paper introduces Target-Entropy Decoding (TED), a novel and theoretically-grounded algorithm for controlling LLM inference. The core contribution—solving for temperature to achieve a target entropy—is elegant, well-formalized, and shown to be practical. The manuscript is exceptionally clear, rigorous, and anticipates key scientific questions.

Specifically:
1.  **Novelty and Significance:** The method provides a deterministic, interpretable control mechanism for uncertainty, a significant departure from heuristic-based or stochastic feedback methods.
2.  **Theoretical Soundness:** The monotonic relationship between temperature and entropy is correctly derived, guaranteeing a unique solution and forming a solid basis for the proposed solver.
3.  **Empirical Rigor:** The experimental design is robust, including multiple models, tasks, strong baselines, and relevant metrics. The inclusion of confidence intervals, detailed microbenchmarks, and insightful ablations (e.g., truncation order) demonstrates a high standard of evidence.
4.  **Reproducibility:** The commitment to releasing code, specific model/data revisions, environment lockfiles, and configuration details sets a high bar for reproducibility and is commendable.

The work is a complete, high-quality contribution suitable for publication in a leading venue. It is well-executed from theory through implementation and evaluation.
