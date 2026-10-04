# Health Answer Safety & Quality Evaluation Framework

I built an evaluation framework that scores AI-generated health answers for accuracy, safety, and whether they escalate appropriately (e.g., recognizing an emergency, or asking for missing information instead of guessing), and validated the scoring against my own hand-labeled examples. I then used an independent model to catch bias in that scoring, and ran an A/B test to see whether adding reference information actually improved answer quality. Finally, I fine-tuned the model on medical Q&A data and used my own evaluation framework to catch a specific, measurable safety regression the fine-tuning introduced — plus a clear explanation of why it happened.

## Notebooks

**`health_eval_framework.ipynb`** — Generates answers to 20 health questions (spanning routine, self-care, emergency, and ambiguous cases) with a local AI model, then has that same model score its own answers for accuracy, safety, and appropriate escalation. It also sets up a blank template for hand-scoring the same answers, so the automated scoring has real human judgment to check against.

**`health_eval_framework_claude_judge.ipynb`** — Uses a second, independent AI model to re-score those same answers, to check whether the first model was just being generous to itself (it was). It also runs an A/B test comparing plain answers against answers given extra reference material, to see if that actually makes them better or safer.

**`finetune_qwen_medquad.ipynb`** — Retrains the AI model on a large set of real medical Q&A data to see if that improves it, then re-runs the same 20 test questions through the retrained model and compares the scores. This is where the project catches the retrained model losing its ability to recognize when it needs to ask for more information before answering.
