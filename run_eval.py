#!/usr/bin/env python3
"""Evaluate GAIA validation results for a specific directory."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "/dfs/share-groups/letrain/zhoujie/AReaL-main")

from customized_areal.tpfc.eval_utils import verify_answer_gaia


async def evaluate_single_file(filepath: Path) -> dict:
    """Evaluate a single GAIA result file."""
    with open(filepath) as f:
        data = json.load(f)

    ground_truth = data.get("ground_truth", "")
    final_boxed_answer = data.get("final_boxed_answer", "")
    task_id = data.get("task_id", "")

    # Handle empty predicted answer
    if not final_boxed_answer or final_boxed_answer.strip() == "":
        print(f"  [{task_id}] Empty final_boxed_answer, marking as INCORRECT")
        is_correct = "INCORRECT"
    else:
        # Use verify_answer_gaia to evaluate
        is_correct = await verify_answer_gaia(ground_truth, final_boxed_answer)
        print(
            f"  [{task_id}] ground_truth='{ground_truth}' vs predicted='{final_boxed_answer}' -> {is_correct}"
        )

    # Update the data with is_correct
    data["is_correct"] = is_correct

    # Write back to file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "task_id": task_id,
        "ground_truth": ground_truth,
        "predicted": final_boxed_answer,
        "is_correct": is_correct,
    }


async def evaluate_directory(base_dir: Path) -> dict:
    """Evaluate all GAIA validation results in a directory."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {base_dir.name}")
    print(f"{'=' * 60}")

    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return {"error": "Directory not found"}

    # Get all JSON files (excluding summary files)
    json_files = [f for f in sorted(base_dir.glob("*.json")) if "summary" not in f.name]
    print(f"Found {len(json_files)} JSON files to evaluate\n")

    results = []
    for filepath in json_files:
        try:
            result = await evaluate_single_file(filepath)
            results.append(result)
        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"] == "CORRECT")
    incorrect = sum(1 for r in results if r["is_correct"] == "INCORRECT")
    not_attempted = sum(1 for r in results if r["is_correct"] == "NOT_ATTEMPTED")

    success_rate = 100 * correct / total if total > 0 else 0

    print(f"\n  Summary for {base_dir.name}:")
    print(
        f"  Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Not Attempted: {not_attempted}"
    )
    print(f"  Success Rate: {correct}/{total} = {success_rate:.2f}%")

    # Save summary to file
    summary = {
        "directory": str(base_dir),
        "model": base_dir.parent.name,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "success_rate": correct / total if total > 0 else 0,
        "success_rate_percent": success_rate,
        "results": results,
    }

    summary_path = base_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    return summary


async def main():
    """Evaluate the specified directory."""
    target_dir = Path(
        "/dfs/share-groups/letrain/zhoujie/AReaL-main/logs/gaia-validation-openrouter/qwen/qwen3-vl-8b-thinking-base_retry_reasoning_fixtool/level_1"
    )
    summary = await evaluate_directory(target_dir)

    # Print final summary
    if "error" not in summary:
        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Model: {summary['model']}")
        print(f"Total: {summary['total']}")
        print(f"Correct: {summary['correct']}")
        print(f"Incorrect: {summary['incorrect']}")
        print(f"Not Attempted: {summary['not_attempted']}")
        print(f"Success Rate: {summary['success_rate_percent']:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
