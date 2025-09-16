Selected idea #5:

5) Target-Entropy Decoding
Summary: Adjust temperature online to follow a specified entropy schedule, aiming to reduce repetition and hallucination while maintaining utility.
For a smart layperson: Keep the model’s uncertainty at a healthy level—neither too random nor too sure—by changing the temperature on the fly. This should avoid boring repetition and overconfident mistakes. The target uncertainty can vary across the response.
Falsification: Implement feedback control to match a per-step entropy target by tuning temperature; evaluate on open-ended writing and factual QA (e.g., TruthfulQA, HaluEval) for repetition rate, factuality, and human preference at equal token budgets. Compare against fixed temperature and nucleus sampling; ablate different entropy schedules.
Novelty: Treating entropy tracking as a primary decoding objective with closed-loop control is new for LLM inference.
