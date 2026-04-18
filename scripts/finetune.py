"""
Fine-tuning script for on-device tool-calling assistant.
Model: meta-llama/Llama-3.2-1B-Instruct (canonical HF repo, loads in transformers v5)
Target: Google Colab T4 (16 GB VRAM)
Output: LoRA adapter in models/llama-1b-tool-calling/

Run on Colab:
  !pip install -r requirements.txt
  !python scripts/finetune.py
"""

import os
import sys

# ── Colab / GPU dependencies ────────────────────────────────────────────────
# These packages are NOT installed locally — run this script on Google Colab T4.
# Install with:  pip install -r requirements-train.txt
# ────────────────────────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch not found. Run: pip install -r requirements-train.txt")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: datasets not found. Run: pip install -r requirements-train.txt")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
except ImportError:
    sys.exit("ERROR: transformers not found. Run: pip install -r requirements-train.txt")

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    sys.exit("ERROR: peft not found. Run: pip install -r requirements-train.txt")

try:
    from trl import SFTTrainer
except ImportError:
    sys.exit("ERROR: trl not found. Run: pip install -r requirements-train.txt")

# DataCollatorForCompletionOnlyLM moved between TRL versions — try both locations
try:
    from trl import DataCollatorForCompletionOnlyLM          # TRL <= 0.9
except ImportError:
    try:
        from trl.extras.dataset_formatting import DataCollatorForCompletionOnlyLM  # TRL 0.10+
    except ImportError:
        sys.exit(
            "ERROR: DataCollatorForCompletionOnlyLM not found.\n"
            "  Try: pip install 'trl>=0.9,<0.11'"
        )

# ============================================================
# CONFIG
# ============================================================
MODEL_ID      = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH  = "data/train.jsonl"
OUTPUT_DIR    = "models/llama-1b-tool-calling"
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 4
GRAD_ACC      = 4
EPOCHS        = 3
LR            = 2e-4

# ============================================================
# SYSTEM PROMPT  (must match inference.py exactly)
# ============================================================
SYSTEM_PROMPT = (
    'You are a helpful on-device assistant with access to these tools:\n'
    '- weather(location: str, unit: "C"|"F")\n'
    '- calendar(action: "list"|"create", date: "YYYY-MM-DD", title: str?)\n'
    '- convert(value: number, from_unit: str, to_unit: str)\n'
    '- currency(amount: number, from: "ISO3", to: "ISO3")\n'
    '- sql(query: str)\n\n'
    'If a tool is needed, respond ONLY with: '
    '<tool_call>{"tool": "...", "args": {...}}</tool_call>\n'
    'Otherwise, respond in plain natural language. Never refuse a valid tool request.'
)

# ============================================================
# CHAT TEMPLATE FORMATTER
# ============================================================
def format_example(example: dict) -> dict:
    """
    Convert a messages list → a single Llama-3 formatted string.
    Only assistant turns are included in the training loss
    (controlled by DataCollatorForCompletionOnlyLM).
    """
    messages = example["messages"]
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    text = ""
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

    return {"text": text}


# ============================================================
# MAIN
# ============================================================
def main():
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # ── 1. Dataset ──────────────────────────────────────────
    raw = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = raw.map(format_example, remove_columns=raw.column_names)
    splits = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = splits["train"]
    eval_ds  = splits["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── 2. Tokenizer ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 3. 4-bit model ─────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ── 4. LoRA ─────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Training args ────────────────────────────────────
    # NOTE: `evaluation_strategy` was renamed to `eval_strategy` in
    #       transformers 4.45. We use the new name; older versions
    #       accept it via an alias.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",          # replaces deprecated evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=False,   # False: avoids PEFT checkpoint reload issues
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_32bit",
        report_to="none",
        group_by_length=True,
        dataloader_num_workers=0,
    )

    # ── 6. Completion-only masking ──────────────────────────
    # Only compute loss on tokens *after* the assistant header.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ── 7. SFTTrainer ───────────────────────────────────────
    # • `tokenizer` replaced by `processing_class` in TRL ≥ 0.9
    # • `dataset_text_field` replaced by a formatting_func in TRL ≥ 0.9
    # We use formatting_func so it works across TRL versions.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=lambda ex: ex["text"],   # already formatted
        max_seq_length=MAX_SEQ_LEN,
        processing_class=tokenizer,              # TRL ≥ 0.9 name
        args=training_args,
        data_collator=collator,
        packing=False,
    )

    # ── 8. Train ────────────────────────────────────────────
    trainer.train()

    # ── 9. Save adapter ─────────────────────────────────────
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to '{OUTPUT_DIR}'")
    print("Next: run scripts/quantize.py to merge + export to GGUF")


if __name__ == "__main__":
    main()
