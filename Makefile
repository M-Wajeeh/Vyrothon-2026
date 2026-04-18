# ─────────────────────────────────────────────────────────────────────────────
#  Vyrothon — On-Device Tool-Calling Assistant
#  Usage: make help
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install data train quantize eval demo clean all

## ── Default target ──────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Vyrothon — On-Device Tool-Calling Assistant"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install    Install all dependencies (requirements.txt)"
	@echo "  make data       Generate synthetic training data"
	@echo "  make train      Fine-tune on Colab T4  (install deps first)"
	@echo "  make quantize   Merge LoRA adapter + export GGUF Q4_K_S"
	@echo "  make eval       Score against public_test.jsonl"
	@echo "  make demo       Launch Streamlit chatbot on localhost:8501"
	@echo "  make all        Full pipeline: data → train → quantize → eval"
	@echo "  make clean      Remove generated model files and __pycache__"
	@echo ""

## ── Install ─────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

## ── Full pipeline (Colab T4) ─────────────────────────────────────────────────
all: data train quantize eval

## ── Step 1: Synthetic data ───────────────────────────────────────────────────
data:
	python scripts/generate_data.py

## ── Step 2: Fine-tune ────────────────────────────────────────────────────────
train:
	pip install -r requirements.txt
	python scripts/finetune.py

## ── Step 3: Quantize ─────────────────────────────────────────────────────────
quantize:
	python scripts/quantize.py

## ── Step 4: Evaluate ─────────────────────────────────────────────────────────
eval:
	python scripts/evaluate.py

## ── Step 5: Streamlit demo ───────────────────────────────────────────────────
demo:
	pip install -r requirements.txt
	streamlit run app.py

## ── Utility: clean build artefacts ──────────────────────────────────────────
clean:
	@echo "Removing generated models and cache..."
	-rm -rf models/merged models/llama-1b-tool-calling models/model-q4_k_s.gguf
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done. Training data in data/ is preserved."
