Selected idea #3:

3) Cross-Prompt KV Projection Warm-Start
Summary: For semantically similar prompts, a linear projection from one prompt’s KV cache can warm-start another’s cache to reduce initial latency without harming accuracy.
For a smart layperson: If you ask two very similar questions, the model’s “mental state” should also be similar. A simple mapping can reuse the earlier state to start the next answer faster.
Falsification: Build a small library of prefixes and a linear projector from prefix embeddings to KV caches (fit on held-out pairs). Warm-start decoding for similar prompts and compare first-token latency and output quality vs. cold start on small models; failure to improve latency or quality indicates falsification.
Novelty: This proposes reusing and projecting entire KV caches across prompts via a learned linear map, rather than caching logits or embeddings alone.
