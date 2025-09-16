Selected idea #3:

3) Cross-Query KV Segment Reuse for Template-Heavy Agents
Summary: Reuse and splice cached key–value (KV) segments across similar prompt fragments to reduce latency and stabilize outputs in iterative agent loops.
For a smart layperson: Agents often repeat similar prompt pieces (“think, call tool, reflect”). We can cache the model’s internal state for these pieces and reuse them next time, saving work and making behavior steadier.
Falsification: Build a ReAct-style agent with calculator/search tools on HotpotQA andToolBench-like tasks; implement shingled MinHash matching over tokenized prefixes to reuse KV blocks; measure wall-clock latency, tokens, and success versus no-reuse and naive prompt caching.
Novelty: Introduces content-similarity-driven KV grafting across distinct prompts at inference, beyond standard exact-prefix caching.
