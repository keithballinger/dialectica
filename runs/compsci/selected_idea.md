Selected idea #4:

4) Capacity Law for Cache Side-Channel Attacks
Summary: Given a fixed leakage model with secret-dependent memory accesses, the key-recovery success rate of last-level cache timing attacks is determined by an empirically measurable channel capacity that depends only on noise and eviction dynamics.
For a smart layperson: Attacks that read secrets through timing act like sending messages over a noisy line; this theory says the attack’s success depends only on how much information that line can carry, which we can measure, not on idiosyncrasies of the victim code.
Falsification: On multiple hardware platforms, measure cache side-channel capacity via stimulus–response experiments, then run standard key-recovery attacks across diverse victim implementations with the same leakage model; if success rates systematically exceed or fall short of what the measured capacity predicts, the theory is false.
Novelty: Provides a quantitative mapping from measured microarchitectural capacity to attack success, replacing qualitative vulnerability assessments.
