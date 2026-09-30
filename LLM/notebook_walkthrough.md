# Step-by-Step Walkthrough: finetune_llm_paid_search.ipynb

This guide explains every cell in plain English — what it does, why it exists,
and what you should expect to see.

---

## The Big Picture

The notebook answers one question:
> "Can we fine-tune a small open-source LLM to look at a keyword's performance
> stats (CTR, CVR, ROAS, Quality Score, etc.) and output a concrete bid
> recommendation with written rationale?"

The full pipeline is:

```
Synthetic keyword data
        ↓
Convert each row into instruction text  (Prompt Engineering)
        ↓
Split into train / val sets             (Dataset Prep)
        ↓
Load Mistral-7B in 4-bit               (QLoRA model loading)
        ↓
Wrap with LoRA adapters                 (PEFT)
        ↓
Fine-tune with SFTTrainer              (Training)
        ↓
Evaluate perplexity + sample outputs   (Evaluation)
        ↓
Expose as get_bid_recommendation()     (Inference)
        ↓
Save adapter / reload / merge          (Deployment)
```

---

## Cell-by-Cell Breakdown

---

### Cell 00–01 — Markdown: Overview

Just documentation. Describes:
- What the notebook builds
- Which libraries are used and why
- The expected outcome

Nothing runs here.

---

### Cell 02 — pip installs

```python
!pip install transformers peft trl bitsandbytes accelerate ...
```

**What it does:** Installs all Python packages the notebook needs.

**Key packages and why each matters:**

| Package | Role |
|---------|------|
| `transformers` | Load and run the base LLM (Mistral-7B) |
| `peft` | Add LoRA adapter layers on top of the frozen base model |
| `trl` | `SFTTrainer` — handles the supervised fine-tuning loop |
| `bitsandbytes >=0.49.2` | 4-bit NF4 quantization — lets the 7B model fit in ~5 GB VRAM |
| `accelerate` | Handles GPU device placement and mixed precision |
| `datasets` | HuggingFace Dataset objects for train/val |

**Important:** After running this cell on a fresh environment, restart the
kernel before proceeding.

---

### Cell 03 — Markdown: Imports section header

Documentation only.

---

### Cell 04 — Imports + seed

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, ...
from peft import LoraConfig, get_peft_model, ...
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
...
SEED = 42
```

**What it does:** Imports everything and sets random seeds for reproducibility
across Python, NumPy, and PyTorch.

**Expected output:**
```
PyTorch version : 2.x.x+cu130
CUDA available  : True
GPU             : NVIDIA GeForce RTX 3090
VRAM            : 25.8 GB
```

If CUDA shows False, the GPU environment isn't set up correctly.

---

### Cell 05 — Markdown: Dataset section header

Documentation only. Explains why synthetic data is used (no real advertiser
data is needed to learn the pipeline).

---

### Cell 06 — Synthetic dataset generation

```python
def generate_keyword_row(keyword, campaign) -> dict:
    ...

df = pd.DataFrame(rows)   # 500 rows
```

**What it does:** Builds a fake but realistic paid search dataset.

Three campaign types are created with different performance profiles:

```
Brand      → 160 rows   high CTR, high CVR, high QS, low CPC
Non-Brand  → 220 rows   medium CTR, medium CVR, medium QS, medium CPC
Competitor → 120 rows   low CTR, low CVR, low QS, high CPC
```

For each row, the function samples:
- `bid_usd` from a uniform range per campaign type
- `impressions`, `clicks` = impressions × CTR
- `conversions` = clicks × CVR
- `revenue` = conversions × average order value
- `roas` = revenue / spend
- `quality_score` from a per-type range

**Expected output:**
```
Dataset shape   : (500, 12)
Campaign counts :
Non-Brand     220
Brand         160
Competitor    120
```

---

### Cell 07 — Descriptive statistics

```python
df[["bid_usd", "ctr", "cvr", "roas", "quality_score"]].describe()
```

**What it does:** Prints summary stats (mean, std, min, max, percentiles) for
the five key numeric columns. Used to sanity-check the distributions look
realistic before building prompts.

---

### Cell 08 — Distribution plots

```python
fig, axes = plt.subplots(2, 2, ...)
# histograms of ROAS, CVR, CTR, Quality Score by campaign type
```

**What it does:** Renders a 2×2 grid of histograms, one per metric, with each
campaign type shown as an overlapping colored histogram. Confirms that Brand,
Non-Brand, and Competitor keywords have visually distinct distributions —
important because the model needs this signal to learn different rules per
campaign type.

---

### Cell 09 — Markdown: Prompt engineering section header

Explains the Mistral-Instruct chat format:
```
<s>[INST] <<SYS>>
{system prompt}
<</SYS>>

{user message} [/INST] {assistant response}</s>
```

This is the template the model was pre-trained on, so we must follow it exactly.

---

### Cell 10 — Hand-crafted example responses

```python
EXAMPLE_RESPONSES = {
    "high_roas_brand":    "Recommendation: INCREASE bid to $1.80...",
    "low_roas_nonbrand":  "Recommendation: DECREASE bid to $0.90...",
    "zero_conversions":   "Recommendation: PAUSE keyword...",
    "competitor_moderate": "Recommendation: MAINTAIN bid at $1.50...",
}
```

**What it does:** Defines 4 manually written gold-standard responses to show
what ideal output looks like. These are NOT used as training data directly.
They are a reference for the developer to calibrate the format, and they anchor
the quality level the rule-based label generator should match.

**Key insight:** In a real system, you would replace the rule-based generator
with either these human-written labels (scaled by experts) or a strong teacher
model (e.g. Claude Opus / GPT-4) generating labels in this exact format.

---

### Cell 11 — Prompt engineering: the core cell

This is the most important cell in the notebook. It defines three things:

**1. `SYSTEM_PROMPT`** — Injected at the start of every training example.
Tells the model its role, the task, and the target ROAS per campaign type.
This is the "persona" the model learns to embody.

**2. `generate_assistant_response(row)`** — The rule-based label generator.
This function reads a row's metrics and produces the target text the model
learns to imitate:

```
if conversions == 0 and clicks > 50  → PAUSE
elif roas >= 1.5 × target and qs >= 7 → INCREASE bid by 35%
elif roas >= 0.9 × target             → MAINTAIN (or small INCREASE)
else                                  → DECREASE bid
```

The output text always follows the same structure:
```
Recommendation: {ACTION} bid to ${new_bid} (currently ${old_bid}).

Rationale:
- {metric-specific bullet 1}
- {metric-specific bullet 2}
- ...

Watch: {monitoring instruction}
```

**3. `row_to_prompt(row)`** — Assembles the complete training string:

```
<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

Keyword       : acme shoes
Match Type    : Exact
Campaign      : Brand
Current Bid   : $0.85
Impressions   : 3,420
Clicks        : 821
Conversions   : 147
Revenue       : $14,112.00
CTR           : 24.00%
CVR           : 17.90%
ROAS          : 19.72
Quality Score : 9/10 [/INST] Recommendation: INCREASE bid to $1.15...</s>
```

Finally: `df["prompt"] = df.apply(row_to_prompt, axis=1)` applies this to all
500 rows to produce the full dataset of training examples.

---

### Cell 12 — Show 3 varied prompt examples

Prints 3 randomly selected complete prompts (rows 50, 150, 400) so you can
visually inspect the quality and diversity of training examples before moving
on.

---

### Cell 13 — Markdown: Dataset prep section header

Documentation for the next two cells.

---

### Cell 14 — Train/val split and HuggingFace Dataset

```python
train_df, val_df = train_test_split(df[["prompt"]], test_size=0.15)
# → 425 train, 75 validation

dataset_dict = DatasetDict({
    "train":      Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
})
```

**What it does:** Splits the 500 prompts 85/15 and wraps them in HuggingFace
`Dataset` objects — the format `SFTTrainer` expects.

Only the `"prompt"` column is kept. The model will see these as raw text strings.

---

### Cell 15 — Token length analysis

```python
df["approx_tokens"] = df["prompt"].apply(estimate_token_length)
MAX_SEQ_LENGTH = 512
```

**What it does:** Estimates how many tokens each prompt is (using a word-count
heuristic before the real tokenizer is loaded). Sets `MAX_SEQ_LENGTH = 512`
based on the distribution — this value is used to truncate inputs during
training. The plot shows almost all prompts are ~260–280 tokens, well under 512.

**Why this matters:** Longer sequences use more VRAM quadratically (attention
is O(n²)). Knowing the distribution lets you set the tightest safe limit.

---

### Cell 16 — Markdown: QLoRA section header

Explains the memory math:
- Mistral-7B in fp32 = ~28 GB — doesn't fit on a single 3090
- Mistral-7B in 4-bit NF4 = ~4.5 GB — fits easily, with room for gradients

---

### Cell 17 — Load model in 4-bit + tokenizer

```python
USE_SMALL_MODEL = False   # True = TinyLlama (~2 GB), False = Mistral-7B (~14 GB)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for weight distributions
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # Quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, ...)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token   # Mistral has no pad token by default
```

**What it does step by step:**
1. Defines the 4-bit quantization config (NF4 + double quant + bf16 compute)
2. Downloads and loads the model weights, quantizing them on the fly
3. Loads the tokenizer and sets the padding token

**What "4-bit NF4" means:** The model weights are stored as 4-bit integers
(16 possible values) mapped to a NormalFloat distribution that matches the
typical bell-curve shape of neural network weights. During the forward pass,
they are dequantized to bf16 for computation.

**Expected output:**
```
Loading model: mistralai/Mistral-7B-Instruct-v0.2
Model loaded. Device map: {'': 0}
Vocabulary size : 32,000
```

---

### Cell 18 — Markdown: LoRA section header

Explains the math: instead of updating the full weight matrix W (7B params),
LoRA adds two small matrices A and B where:

```
W_new = W_frozen + α/r · (B × A)
```

Where `r` = rank (how many dimensions) and `α` = scaling factor.
Only A and B are trained — roughly 0.1% of total parameters.

---

### Cell 19 — Apply LoRA to the model

```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,                    # rank — size of the adapter matrices
    lora_alpha=32,           # scaling = alpha/r = 2.0
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
```

**What it does step by step:**
1. `prepare_model_for_kbit_training` — casts LayerNorm layers to fp32 for
   training stability, enables gradient checkpointing (trades compute for VRAM)
2. `LoraConfig` — specifies where and how to insert LoRA matrices
3. `get_peft_model` — injects the LoRA matrices into the 7 attention/MLP
   module types listed in `target_modules` and freezes all base model weights

**Expected output:**
```
Trainable parameters :    83,886,080  (1.1% of total)
Frozen parameters    : 7,158,525,952
Total parameters     : 7,242,412,032
```

Only the ~84M LoRA parameters will receive gradient updates. The 7B base model
weights stay frozen and unchanged.

---

### Cell 20 — Visualise LoRA adapter architecture

Lists all trainable LoRA tensor names and their sizes. Useful for confirming
LoRA was injected into the right layers. Also shows a bar chart of parameter
counts per layer.

---

### Cell 21 — Markdown: Training section header

Explains what SFTTrainer does differently from a vanilla HuggingFace Trainer:
- Handles text formatting automatically
- Supports completion-only loss masking
- Integrates with PEFT natively

---

### Cell 22 — TrainingArguments

```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16
    optim="paged_adamw_32bit",       # memory-efficient optimizer from bitsandbytes
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    ...
)
```

**What each key argument does:**

| Argument | Value | Why |
|----------|-------|-----|
| `gradient_accumulation_steps=4` | 4 | Simulates batch size 16 with only 4 samples in memory at once |
| `optim="paged_adamw_32bit"` | — | Adam optimizer with paged memory — prevents OOM from optimizer states |
| `learning_rate=2e-4` | 0.0002 | Standard LoRA LR — higher than full fine-tuning because adapters are small |
| `lr_scheduler_type="cosine"` | — | LR decays smoothly to near-zero by end of training |
| `warmup_ratio=0.03` | 3% | LR ramps up for first 3% of steps to avoid instability at the start |
| `load_best_model_at_end=True` | — | After training, reverts to the checkpoint with lowest eval loss |

---

### Cell 23 — SFTTrainer setup

```python
RESPONSE_TEMPLATE = " [/INST] "

collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=collator,
    dataset_text_field="prompt",
    max_seq_length=MAX_SEQ_LENGTH,
    ...
)
```

**The most important thing here: completion-only loss masking.**

Without the `DataCollatorForCompletionOnlyLM`, the model would compute loss
on every token — including the system prompt and user message. That's wasteful:
we don't want the model to "learn" the fixed system prompt text.

With the collator, loss is only computed on tokens **after** `" [/INST] "`.
So the model only learns to predict the bid recommendation text — not the
question being asked.

Visually:
```
<s>[INST] <<SYS>>...{user message} [/INST] Recommendation: INCREASE...
─────────────────────────────────────────────────────────────────────────
[  no loss computed here — prompt tokens masked to -100  ] [ loss here ]
```

---

### Cell 24 — trainer.train()

```python
train_result = trainer.train()
```

**What happens during training:**

For each step:
1. A batch of 4 prompts is tokenized and padded
2. Forward pass: model predicts next token for each position
3. Loss is computed only on the response tokens (after `[/INST]`)
4. Backward pass: gradients flow only through the LoRA matrices (base model
   weights have `requires_grad=False`)
5. AdamW updates the LoRA A and B matrices
6. Every 4 steps, the optimizer actually takes a step (gradient accumulation)

After each epoch, validation loss is computed on the 75 held-out examples.
The checkpoint with the lowest validation loss is saved.

**Expected output (logged every 10 steps):**
```
Step  10 | loss: 2.41 | lr: 0.000021
Step  20 | loss: 1.87 | lr: 0.000063
...
Epoch 1 | eval_loss: 0.94
...
Epoch 3 | eval_loss: 0.71
```

Loss should decrease from ~2.4 → ~0.7 over 3 epochs.

---

### Cell 25 — Training curve plot

```python
train_losses = [(l["step"], l["loss"]) for l in log_history if "loss" in l]
eval_losses  = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]
```

Plots training and validation loss on the same chart. What to look for:
- Both curves decreasing — good
- Val loss much higher than train loss — overfitting (reduce epochs or add dropout)
- Val loss lower than train loss — very unlikely, check data leakage

---

### Cell 26 — Markdown: Evaluation section header

Explains the two evaluation methods used: perplexity and sample generation.

---

### Cell 27 — `compute_perplexity()` function

```python
def compute_perplexity(model, tokenizer, texts, max_length=512):
    # runs model forward on each text, reads the loss, converts to perplexity
    return math.exp(total_loss / total_tokens)
```

**What perplexity means:** If perplexity = 2.0, the model is on average
choosing between 2 equally likely options at each token. Lower = more confident
= better fit to the validation data.

Typical values:
- Untrained 7B model on this domain: ~40–80
- After fine-tuning: ~2–5

Limitations: perplexity measures how well the model fits the training
distribution — not whether the recommendations are actually correct or
profitable. You need real A/B tests for that.

---

### Cell 28 — `generate_response()` function

```python
def generate_response(model, tokenizer, prompt_text, max_new_tokens=300, temperature=0.1):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)
    # strip the prompt from the output, return only the generated text
    return full_text[len(prompt_text):].strip()
```

Low temperature (0.1) = near-deterministic output. For bid recommendations
you want consistent, not creative, outputs.

---

### Cell 29 — Markdown: Inference section header

Describes the wrapper function and shows 5 held-out keyword examples.

---

### Cell 30 — `get_bid_recommendation()` function

```python
def get_bid_recommendation(keyword_stats: dict) -> str:
    row = pd.Series(keyword_stats)
    prompt = row_to_inference_prompt(row)   # same format but WITHOUT the answer
    return generate_response(model, tokenizer, prompt)
```

This is the production-facing interface. You pass a dictionary of keyword
metrics, get back a recommendation string. Five example outputs are shown
on held-out keywords not seen during training.

---

### Cell 31 — Markdown: Save/load section header

Explains that only the adapter weights (~50–200 MB) are saved, not the full
7B base model (~14 GB). The base model can be re-downloaded from HuggingFace
whenever needed.

---

### Cell 32 — Save adapter

```python
ADAPTER_DIR = "./paid_search_lora_adapter/final_adapter"
model.save_pretrained(ADAPTER_DIR)      # saves adapter_config.json + adapter weights
tokenizer.save_pretrained(ADAPTER_DIR)  # saves tokenizer files
```

What gets saved:
```
final_adapter/
├── adapter_config.json        ← LoRA hyperparameters (r, alpha, target_modules)
├── adapter_model.safetensors  ← the actual LoRA weight matrices (~160 MB)
└── tokenizer files            ← vocab, tokenizer config
```

---

### Cell 33 — Reload adapter

```python
base_model_reload = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, ...)
model_reloaded = PeftModel.from_pretrained(base_model_reload, ADAPTER_DIR)
```

Shows how to reconstruct the fine-tuned model from scratch:
1. Load base model again from HuggingFace (quantized)
2. Load LoRA adapter on top
3. The combined model behaves identically to the trained model

---

### Cell 34 — Merge adapter into base model (optional)

```python
# Commented out by default — only run if you have ~28 GB VRAM available
merged_model = model_with_adapter.merge_and_unload()
merged_model.save_pretrained(MERGE_OUTPUT_DIR)
```

**What merging does:** Mathematically folds the LoRA matrices back into the
base model weights:

```
W_final = W_frozen + (B × A) × (alpha / r)
```

The result is a standard model with no PEFT dependency — faster at inference
and easier to serve with vLLM or similar frameworks. Requires loading the base
model in fp16 (not 4-bit), so needs more VRAM.

---

### Cell 35 — Markdown: Production considerations

Covers what needs to change to go from notebook → production:

| Topic | Consideration |
|-------|--------------|
| **Serving** | Use vLLM or TGI instead of HuggingFace generate — 10–50x throughput |
| **Batching** | Process full keyword portfolio overnight in batches, not one-by-one |
| **Latency** | Merging LoRA + quantized serving gets to ~50–200ms per keyword |
| **Retraining** | Retrain monthly as new campaign data accumulates |
| **Monitoring** | Track if real bids follow recommendations and compare ROAS pre/post |
| **A/B testing** | Split keywords into control (existing rules) and treatment (LLM recs) |

---

### Cell 36 — Markdown: Business impact

How to measure whether the model is actually working:

- **Primary metric:** ROAS improvement in the A/B test treatment group
- **Secondary:** Reduction in wasted spend (spend on zero-conversion keywords)
- **Efficiency:** Time saved vs. human bid managers reviewing keywords manually
- **Coverage:** % of long-tail keywords receiving a recommendation vs. ignored

---

## Summary: What the Notebook Actually Teaches

```
Data side:
  Tabular keyword data → text prompts → HuggingFace Dataset

Model side:
  7B frozen base model → 4-bit quantized (fits in 5 GB VRAM)
                       → LoRA adapters injected (only 84M params train)
                       → SFT on 425 examples
                       → 75-example validation

Training trick:
  Completion-only loss masking → model only learns to predict the answer,
  not re-learn the question format

Output:
  Adapter weights (~160 MB) that sit on top of the base model
  A callable function: keyword_stats dict → recommendation string
```

The key design choices — QLoRA for memory efficiency, LoRA for parameter
efficiency, completion-only masking for training efficiency, and rule-based
labels as a starting point — are each independently replaceable as the system
matures.
