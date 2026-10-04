# mle_sandbox

- **`LLM/`** — Fine-tuning an LLM to generate bid recommendations for paid search campaigns, using a synthetic keyword-performance dataset. Covers QLoRA fine-tuning, evaluation, and a causal backtest to check whether the model's recommendations would actually improve campaign performance.
- **`health_search_eval/`** — An evaluation framework for AI-generated health answers, scoring them for accuracy, safety, and appropriate escalation. Covers bias-checking the scorer with an independent judge, an A/B test on answer grounding, and a LoRA fine-tune with a regression check against the same evaluation harness.
