"""
eval_harness_contract.py — Exact interface the grader will call.
DO NOT MODIFY this contract. Your inference.py must conform to it.
"""
import json
import hashlib
import time
import sys

def load_test_set(path: str) -> list[dict]:
    """Load test examples from JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def score_response(response: str, expected: dict | None) -> float:
    """
    Score a single response.
    Returns: +1.0, +0.5, 0.0, or -0.5
    """
    has_tool_call = "<tool_call>" in response and "</tool_call>" in response
    
    # Refusal case
    if expected is None:
        if has_tool_call:
            return -0.5  # Emitted tool call when refusal was correct
        return 1.0  # Correct refusal
    
    # Tool call expected
    if not has_tool_call:
        return 0.0  # No tool call found
    
    try:
        json_str = response.split("<tool_call>")[1].split("</tool_call>")[0]
        call = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return 0.0  # Malformed JSON
    
    if call.get("tool") != expected["tool"]:
        return 0.0  # Wrong tool
    
    # Check args
    expected_args = expected.get("args", {})
    actual_args = call.get("args", {})
    
    all_match = True
    for key, expected_val in expected_args.items():
        actual_val = actual_args.get(key)
        if actual_val is None:
            all_match = False
            continue
        # Numerical tolerance ±1%
        if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
            if expected_val == 0:
                if actual_val != 0:
                    all_match = False
            elif abs(actual_val - expected_val) / abs(expected_val) > 0.01:
                all_match = False
        elif str(actual_val).lower() != str(expected_val).lower():
            all_match = False
    
    if all_match:
        return 1.0  # Exact match
    else:
        return 0.5  # Correct tool, wrong args

def run_evaluation(inference_module, test_path: str):
    """
    Run the full evaluation.
    inference_module must expose: def run(prompt: str, history: list[dict]) -> str
    """
    examples = load_test_set(test_path)
    total_score = 0.0
    total_time = 0.0
    results = []
    
    for ex in examples:
        messages = ex["messages"]
        # Last user message is the prompt
        prompt = messages[-1]["content"]
        # Everything before is history
        history = messages[:-1] if len(messages) > 1 else []
        
        expected = ex.get("expected")
        
        start = time.time()
        response = inference_module.run(prompt, history)
        elapsed = time.time() - start
        total_time += elapsed
        
        score = score_response(response, expected)
        total_score += score
        
        results.append({
            "id": ex.get("id", "?"),
            "prompt": prompt,
            "response": response,
            "expected": expected,
            "score": score,
            "latency_ms": round(elapsed * 1000, 1)
        })
    
    mean_latency = (total_time / len(examples)) * 1000
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {total_score:.1f}/{len(examples)} points")
    print(f"Mean latency: {mean_latency:.1f} ms/turn")
    print(f"{'='*60}")
    
    for r in results:
        status = "✅" if r["score"] == 1.0 else "⚠️" if r["score"] == 0.5 else "❌"
        print(f"{status} [{r['score']:+.1f}] id={r['id']} | {r['latency_ms']:.0f}ms | {r['prompt'][:50]}...")
    
    return {
        "total_score": total_score,
        "max_score": len(examples),
        "mean_latency_ms": mean_latency,
        "results": results,
    }

if __name__ == "__main__":
    # Usage: python eval_harness_contract.py [test_file]
    test_file = sys.argv[1] if len(sys.argv) > 1 else "starter/public_test.jsonl"
    
    import inference
    run_evaluation(inference, test_file)
