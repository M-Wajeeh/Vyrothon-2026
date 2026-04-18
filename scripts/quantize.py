"""
Quantization script: Merge LoRA adapter into base model, then export to GGUF.
Target size: <= 500 MB (Q4_K_S quantization for Llama-3.2-1B)

Steps:
  1. Load base model + adapter, merge weights
  2. Save merged model in HF format
  3. Call llama.cpp convert script to produce GGUF at Q4_K_S

Requirements (install in Colab):
  !pip install peft transformers bitsandbytes
  !git clone https://github.com/ggerganov/llama.cpp /content/llama.cpp
  !pip install -r /content/llama.cpp/requirements.txt
"""

import os
import subprocess
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# CONFIG — adjust paths as needed
# ============================================================
BASE_MODEL_ID  = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR    = "models/llama-1b-tool-calling"
MERGED_DIR     = "models/merged"
GGUF_PATH      = "models/model-q4_k_s.gguf"
LLAMA_CPP_DIR  = "/content/llama.cpp"          # adjust for local run
QUANT_TYPE     = "Q4_K_S"                       # ~490 MB — right under 500 MB gate
TARGET_SIZE_MB = 500

def merge_and_save():
    """Load base + LoRA adapter, merge weights, save merged model."""
    print(f"Loading base model: {BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",  # Merge on CPU to avoid GPU OOM
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print(f"Loading adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()
    
    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"Saving merged model to: {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print("Merge complete.")

def convert_to_gguf():
    """Convert merged HF model to GGUF using llama.cpp."""
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Try older script name
        convert_script = os.path.join(LLAMA_CPP_DIR, "convert-hf-to-gguf.py")
    
    if not os.path.exists(convert_script):
        print(f"ERROR: llama.cpp convert script not found at {LLAMA_CPP_DIR}")
        print("Clone it with: git clone https://github.com/ggerganov/llama.cpp")
        return False
    
    fp16_gguf = GGUF_PATH.replace(f"-{QUANT_TYPE.lower()}", "-fp16").replace(".gguf", "-fp16.gguf")
    
    print("Step 1: Converting HF model to FP16 GGUF...")
    cmd_convert = [
        "python", convert_script,
        MERGED_DIR,
        "--outfile", fp16_gguf,
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion failed:\n{result.stderr}")
        return False
    print(f"FP16 GGUF saved: {fp16_gguf}")
    
    print(f"Step 2: Quantizing to {QUANT_TYPE}...")
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(LLAMA_CPP_DIR, "quantize")  # older name
    
    if not os.path.exists(quantize_bin):
        print("ERROR: llama-quantize binary not found. Build llama.cpp first:")
        print("  cd /content/llama.cpp && make -j4")
        return False
    
    os.makedirs(os.path.dirname(GGUF_PATH), exist_ok=True)
    cmd_quant = [quantize_bin, fp16_gguf, GGUF_PATH, QUANT_TYPE]
    result = subprocess.run(cmd_quant, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Quantization failed:\n{result.stderr}")
        return False
    
    size_mb = os.path.getsize(GGUF_PATH) / (1024 ** 2)
    print(f"Quantized GGUF saved: {GGUF_PATH}")
    print(f"Size: {size_mb:.1f} MB (limit: {TARGET_SIZE_MB} MB)")
    
    if size_mb > TARGET_SIZE_MB:
        print(f"WARNING: {size_mb:.1f} MB exceeds {TARGET_SIZE_MB} MB hard gate!")
        print("Consider using Q3_K_M for further size reduction.")
    else:
        print(f"✅ Size gate PASSED ({size_mb:.1f} MB <= {TARGET_SIZE_MB} MB)")
    
    # Cleanup FP16 intermediate
    if os.path.exists(fp16_gguf):
        os.remove(fp16_gguf)
    
    return True

def main():
    print("=" * 60)
    print("Quantization Pipeline: LoRA adapter → GGUF")
    print("=" * 60)
    
    if not os.path.exists(ADAPTER_DIR):
        print(f"ERROR: Adapter not found at '{ADAPTER_DIR}'")
        print("Run scripts/finetune.py first.")
        return
    
    merge_and_save()
    success = convert_to_gguf()
    
    if success:
        print("\n✅ Quantization pipeline complete!")
        print(f"Model ready at: {GGUF_PATH}")
        print(f"Set MODEL_PATH env var to use it:\n  export MODEL_PATH={GGUF_PATH}")
    else:
        print("\n❌ Quantization failed. See errors above.")

if __name__ == "__main__":
    main()
