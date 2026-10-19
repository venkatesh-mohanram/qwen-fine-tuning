Qwen2.5 LoRA Lab — IEEE iGET 2026

Getting started
- Open Notebook_2_iGET_Qwen_LoRA_Adaptive_CPU_GPU.ipynb.
- Keep all five supplied JSONL files in the notebook's working directory.
  In a hosted environment, upload the notebook and data files together.
- Use Python 3.10+ and a dedicated environment; see section 0 for kernel setup.
- Install the libraries, run %pip check, restart the kernel, then continue from section 1.
- Hardware detection selects CUDA GPU settings when available; otherwise it uses CPU.

Learning objectives
- Inspect tokens, special tokens, chat boundaries, embeddings, and model architecture.
- Identify frozen base weights and trainable LoRA A/B matrices.
- Train an adapter and compare weights and answers before and after fine-tuning.
- Evaluate exact recall and paraphrased questions, then save and reload the adapter.

Default memorization experiment (DEMO_MEMORIZATION=True in section 6)
- Four existing training examples: dates, location, workshop deadline, poster date.
- GPU: 50 epochs, batch size 2; CPU: 25 epochs, batch size 1.
- Both configurations use 100 optimizer steps with gradient accumulation 1.
- Learning rate 5e-4, constant schedule, zero weight decay, seed 42.
- LoRA rank 16, alpha 32, dropout 0; target modules q_proj and v_proj.

Full-data experiment (DEMO_MEMORIZATION=False)
- GPU: 38 training examples, 8 validation examples, 5 epochs, batch size 2.
- CPU: 16 training examples, 4 validation examples, 2 epochs, batch size 1.
- Learning rate 2e-4, linear schedule, LoRA rank 8, alpha 16, dropout 0.05.
- The same 11-question test set is available in both hardware modes.

Shared training settings
- GPU uses float16; CPU uses float32.
- Standard adamw_torch optimizer and completion-only negative log-likelihood loss.
- Initial sequence budgets: 256 tokens on GPU, 128 on CPU. The training limit is
  increased when needed to preserve all selected training and validation sequences.
- Inference uses greedy decoding with the same tokenizer and chat template as training.

Notebook guide
- Section 3.1: validate all five data files and inspect duplicate/overlap counts.
- Sections 4.1-4.5: inspect tokenization, embeddings, architecture, and sequence lengths.
- Section 6: select the training mode and capture base-model answers.
- Sections 8-9: attach LoRA and inspect frozen W and trainable A/B matrices.
- Sections 10-11: configure training, train, and compare parameter changes and recall.
- Sections 12-17: plot loss, save the adapter, evaluate answers, and reload the adapter.
- Sections 18-19: explore missing information and further experiments.

Interpreting the results
- An epoch is one pass through the training dataset. Additional epochs can improve
  memorization but do not guarantee exact generation and can also worsen results.
- Exact matching ignores whitespace only; a different answer may still be correct.
- DEMO_READY and RELOADED_DEMO_READY indicate whether all four references match.
- Test questions cover overlapping facts with different wording. Four-example mode
  does not train on every fact in the test set.
- Keyword accuracy is a heuristic; inspect generated answers as well as scores.
- Source metadata records provenance and is excluded from training. CPU datasets
  are subsets of their GPU counterparts. Consult source pages for current event facts.

Saving and repeating experiments
- Adapter directories include the hardware and memorization/full-data mode.
- demo_rehearsal.json stores training settings and the four-question evaluation.
- Copy an adapter directory elsewhere to retain it before rerunning the same mode.
- Restart the kernel and rerun the notebook in order when changing training settings.
- Compare loss, exact recall, and paraphrase scores across runs.


Dependency compatibility
- The notebook uses Transformers 4.57.6, TRL 0.29.0, and PEFT 0.18.0 APIs.
- Transformers 4.x requires huggingface-hub<1. Gradio 6.29 and Diffusers 0.40
  require Hub 1.x and cannot coexist with this stack in one environment.
- Use a separate kernel/runtime for the lab; do not upgrade Hub alone to resolve
  these conflicts. A fresh hosted runtime may still contain incompatible packages.
- %pip targets the active notebook kernel. %pip check verifies installed requirements.
- Restart after installation; restart alone does not fix dependency conflicts.


Optional torchao troubleshooting
- Section 1 checks installed and already loaded Transformers/PEFT/TRL versions
  before model loading, then probes PEFT's optional torchao version check.
- A torchao>=0.16.0 error differs from the pinned PEFT 0.18.0 check. Rerun the
  section 0 installation, restart the kernel, and resume from section 1.
- This lab uses floating-point LoRA and does not require torchao quantization.
  Do not change LoRA rank/alpha/dropout to address a package-version error.
