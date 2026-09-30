# Related Papers: Fine-Tuning LLMs for Paid Search Optimization

---

## 1. Foundational Methods (techniques used in the notebook)

**LoRA: Low-Rank Adaptation of Large Language Models**
- Hu et al. — ICLR 2022
- arXiv: https://arxiv.org/abs/2106.09685
- Freezes pre-trained weights and injects trainable low-rank matrices into each transformer layer, reducing trainable parameters by up to 10,000x vs. full fine-tuning with no performance loss. The direct basis for the adapter approach used in the notebook.

**QLoRA: Efficient Finetuning of Quantized LLMs**
- Dettmers et al. — NeurIPS 2023
- arXiv: https://arxiv.org/abs/2305.14314
- Combines 4-bit NF4 quantization of the frozen base model with LoRA adapters in bf16, plus Double Quantization and Paged Optimizers. Allows fine-tuning a 65B model on a single 48GB GPU. This is the exact method in the notebook (BitsAndBytesConfig + LoRA).

**InstructGPT: Training Language Models to Follow Instructions with Human Feedback**
- Ouyang et al. (OpenAI) — NeurIPS 2022
- arXiv: https://arxiv.org/abs/2203.02155
- Introduces the 3-stage RLHF pipeline: SFT on demonstrations → reward model → PPO. The notebook implements stage 1 (SFT). A 1.3B InstructGPT model was preferred over untuned 175B GPT-3 by human raters.

**Direct Preference Optimization (DPO)**
- Rafailov et al. (Stanford) — NeurIPS 2023
- arXiv: https://arxiv.org/abs/2305.18290
- Eliminates the reward model and PPO from RLHF by re-parameterizing the objective as a binary cross-entropy loss over preferred vs. rejected response pairs. The natural next step after the notebook's SFT stage to align recommendations with expert preferences.

**FLAN: Finetuned Language Models Are Zero-Shot Learners**
- Wei et al. (Google) — ICLR 2022
- arXiv: https://arxiv.org/abs/2109.01652
- Establishes instruction tuning — fine-tuning on 60+ tasks framed as natural language instructions — as the key technique for generalization. Explains why the notebook's instruction format (system prompt + user stats + assistant response) generalizes to unseen keywords.

**Scaling Instruction-Finetuned Language Models (Flan-T5 / Flan-PaLM)**
- Chung et al. (Google) — JMLR 2024
- arXiv: https://arxiv.org/abs/2210.11416
- Scales instruction fine-tuning across model size, task count, and chain-of-thought data. Flan-PaLM 540B improves over base PaLM by 9.4% on average. Flan-T5 checkpoints are widely used as SFT baselines.

---

## 2. LLMs for Advertising & Paid Search

**Applying Large Language Models to Sponsored Search Advertising**
- Reisenbichler, Reutterer, Schweidel — Marketing Science 2024
- DOI: https://pubsonline.informs.org/doi/10.1287/mksc.2023.0611
- Closest paper to the notebook. Builds a fine-tuned LLM pipeline for SEM ad copy generation and validates it in live field experiments. Results show significantly improved CTR and lower CPC vs. human-written ads.

**Online Advertisements with LLMs: Opportunities and Challenges**
- Feizi et al. (University of Maryland) — 2023
- arXiv: https://arxiv.org/abs/2311.07601
- Decomposes LLM advertising into modification, bidding, prediction, and auction modules. Provides game-theoretic framing for why standard auction mechanisms must be rethought when delivery is generative. Good architectural reference for productionizing the notebook.

**GRAD: Generative Large-Scale Pre-trained Models for Automated Ad Bidding**
- Multiple authors — KDD 2024
- arXiv: https://arxiv.org/abs/2508.02002
- Foundation model pre-trained on large-scale bidding trajectory data with a Mixture-of-Experts action module and causal transformer value estimator. Production-scale version of the notebook's idea — fine-tuned per advertiser for ROAS/CPM constraints.

---

## 3. Bid Optimization as Sequential Decision Making
### (connects to the ltv_offline_rl course)

**Bid Optimization using Maximum Entropy Reinforcement Learning**
- Liu et al. — Neurocomputing 2022
- arXiv: https://arxiv.org/abs/2110.05032
- Formulates paid search bidding as a sequential decision problem and applies Soft Actor-Critic (SAC) with entropy regularization. Directly maps to the MDP framing in the RL/OPE course.

**Offline Reinforcement Learning for Optimizing Production Bidding Policies**
- Korenkevych et al. — KDD 2024
- arXiv: https://arxiv.org/abs/2310.09426
- Trains offline RL agents to optimize parameters of existing heuristic bidding policies (e.g. target CPA rules) rather than replacing them with black-box networks. Bridges the notebook (LLM recommendations) and the ltv_offline offline RL approach. Demonstrated statistically significant production gains.

**AdCraft: An Advanced RL Benchmark for Search Engine Marketing Optimization**
- Gomrokchi et al. (System1) — 2023
- arXiv: https://arxiv.org/abs/2306.11971
- A realistic, stochastic, non-stationary simulation environment for training RL agents on SEM bid and budget management. Useful for generating richer training trajectories for the notebook rather than relying on static heuristic labels.

**AucArena: Evaluating LLM Agents in Auction Settings**
- Chen et al. — 2023
- arXiv: https://arxiv.org/abs/2310.05746
- Multi-agent simulation testing whether LLMs (GPT-4) can do sequential auction bidding — tracking budgets, modeling competitors, adapting strategy across rounds. Shows LLMs exhibit meaningful but imperfect budget management without explicit RL.

**Do LLM Agents Have Regret? Online Learning and Games**
- Multiple authors — 2024
- arXiv: https://arxiv.org/abs/2403.16843
- Formally analyzes whether LLM agents achieve sublinear regret in repeated bidding games. Finds that pure LLM reasoning is insufficient for optimal bidding without algorithmic scaffolding. Provides theoretical grounding for why the notebook's LLM recs should be combined with RL or constrained optimization in production.

---

## 4. Tabular-to-Text (the notebook's core data transformation)

**TabLLM: Few-shot Classification of Tabular Data with Large Language Models**
- Hegselmann et al. (MIT / Microsoft) — AISTATS 2023
- arXiv: https://arxiv.org/abs/2210.10723
- Serializes tabular rows into natural language strings and fine-tunes T0 on the result. Directly studies the same transformation the notebook applies: keyword performance stats (CTR, CVR, ROAS) → text prompt → LLM prediction. Outperforms standard tabular deep learning in few-shot regimes.

**Large Language Models on Tabular Data: Prediction, Generation, and Understanding — A Survey**
- Fang et al. (Amazon / CMU) — 2024
- arXiv: https://arxiv.org/abs/2402.17944
- Comprehensive survey of all LLM methods for tabular data: serialization strategies, fine-tuning approaches, in-context learning. Best single reference for understanding where LLM tabular prediction sits vs. traditional ML (gradient-boosted trees etc.).

**From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with LLMs**
- Multiple authors — 2023
- arXiv: https://arxiv.org/abs/2310.07338
- Reframes tabular prediction as a generative text completion task (generate the target label token conditioned on the serialized row) rather than discriminative classification. Closely aligned with how the notebook structures bid recommendations as text generation.

---

## Reading Order Recommendation

```
Start here (techniques):
  LoRA → QLoRA → FLAN → InstructGPT → DPO

Then domain papers:
  TabLLM → Applying LLMs to Sponsored Search → GRAD

Then connect to ltv_offline:
  AdCraft (env) → Bid Opt with MaxEnt RL → Offline RL for Production Bidding

Advanced / theoretical:
  AucArena → Do LLM Agents Have Regret?
```
