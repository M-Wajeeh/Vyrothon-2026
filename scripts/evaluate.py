"""
evaluate.py — Local evaluation using the grader contract.
Usage:
  python scripts/evaluate.py                       # runs on starter/public_test.jsonl
  python scripts/evaluate.py path/to/test.jsonl    # custom test file
"""
import sys
import os

# Allow running from scripts/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inference
from starter.eval_harness_contract import run_evaluation

if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "starter/public_test.jsonl"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        sys.exit(1)
    run_evaluation(inference, test_file)
