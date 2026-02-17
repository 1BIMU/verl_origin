import os
import argparse
import subprocess
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from verl.utils.reward_score import default_compute_score


DATASET_REGISTRY = {
    "bbh": "data/offline_eval/bbh__all.parquet",
    "humaneval": "data/offline_eval/code__humaneval.parquet",
    "mbpp": "data/offline_eval/code__mbpp.parquet",
    "livecodebench": "data/offline_eval/code__livecodebench.parquet",
    "bigcodebench": "data/offline_eval/code__bigcodebench.parquet",
    "math500": "data/offline_eval/math__math_500.parquet",
    "aime": "data/offline_eval/math__aime_repeated_8x_240.parquet",
}


def get_dataset_path(name):
    if name in DATASET_REGISTRY:
        return os.path.expanduser(DATASET_REGISTRY[name])
    return os.path.expanduser(name)


def run_generation(model_path, data_path, output_path, n_samples, n_gpus, batch_size, temperature, response_length):
    cmd = [
        sys.executable, "-m", "verl.trainer.main_generation",
        f"model.path={model_path}",
        f"data.path={data_path}",
        f"data.output_path={output_path}",
        f"data.n_samples={n_samples}",
        f"data.batch_size={batch_size}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"rollout.temperature={temperature}",
        "rollout.top_p=0.95",
        "rollout.prompt_length=2048",
        f"rollout.response_length={response_length}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_accuracy_eval(data_path, sandbox_url=None):
    df = pd.read_parquet(data_path)
    results_by_source = defaultdict(list)
    all_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        data_source = row.get("data_source", "unknown")
        responses = row.get("responses", [])
        ground_truth = row.get("reward_model", {}).get("ground_truth", "")
        extra_info = row.get("extra_info", {})

        scores = []
        for r in responses:
            try:
                score = default_compute_score(
                    data_source=data_source,
                    solution_str=r,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    sandbox_fusion_url=sandbox_url,
                )
                if isinstance(score, dict):
                    score = score.get("score", 0.0)
                scores.append(float(score))
            except Exception as e:
                print(f"Error scoring {data_source}: {e}")
                scores.append(0.0)

        mean_score = np.mean(scores) if scores else 0.0
        all_scores.append(mean_score)
        results_by_source[data_source].append(mean_score)

    n_samples = len(df.iloc[0]["responses"]) if len(df) > 0 else 0

    print(f"\n=== Results (mean@{n_samples}) ===")
    for source, scores in sorted(results_by_source.items()):
        print(f"{source}: {np.mean(scores):.4f} (n={len(scores)})")

    overall = np.mean(all_scores) if all_scores else 0.0
    print(f"\nOverall: {overall:.4f}")

    return {
        "overall": overall,
        "n_samples": n_samples,
        "by_source": {k: float(np.mean(v)) for k, v in results_by_source.items()},
    }


def run_self_correction_eval(data_path, output_path, model, max_workers, sample_size):
    cmd = [
        sys.executable, "scripts/llm_judge_self_correction.py",
        f"--input_path={data_path}",
        f"--output_path={output_path}",
        f"--model={model}",
        f"--max_workers={max_workers}",
    ]
    if sample_size:
        cmd.append(f"--sample_size={sample_size}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Dataset names or paths")
    parser.add_argument("--k", type=int, default=16, help="Number of samples for mean@k")
    parser.add_argument("--output_dir", type=str, default="./output/eval")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--response_length", type=int, default=2048)
    parser.add_argument("--sandbox_url", type=str, default=None, help="Sandbox URL for code execution")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_accuracy", action="store_true")
    parser.add_argument("--skip_self_correction", action="store_true")
    parser.add_argument("--llm_judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_judge_workers", type=int, default=10)
    parser.add_argument("--llm_judge_sample_size", type=int, default=None)
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Evaluation Pipeline ===")
    print(f"Model: {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"mean@{args.k}")
    print(f"Output: {output_dir}")

    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        data_path = get_dataset_path(dataset_name)
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping")
            continue

        safe_name = dataset_name.replace("/", "_").replace("~", "")
        generation_output = os.path.join(output_dir, f"{safe_name}_generation.parquet")
        self_correction_output = os.path.join(output_dir, f"{safe_name}_self_correction.json")

        if not args.skip_generation:
            print(f"\n--- Generation ---")
            run_generation(
                model_path=args.model_path,
                data_path=data_path,
                output_path=generation_output,
                n_samples=args.k,
                n_gpus=args.n_gpus,
                batch_size=args.batch_size,
                temperature=args.temperature,
                response_length=args.response_length,
            )

        if not args.skip_accuracy:
            print(f"\n--- Accuracy Evaluation ---")
            accuracy_results = run_accuracy_eval(generation_output, sandbox_url=args.sandbox_url)
            all_results[dataset_name] = {"accuracy": accuracy_results}

        if not args.skip_self_correction and os.environ.get("OPENAI_API_KEY"):
            print(f"\n--- Self-Correction Evaluation ---")
            run_self_correction_eval(
                data_path=generation_output,
                output_path=self_correction_output,
                model=args.llm_judge_model,
                max_workers=args.llm_judge_workers,
                sample_size=args.llm_judge_sample_size,
            )

    print(f"\n{'='*50}")
    print("=== Final Summary ===")
    print(f"{'='*50}")
    for dataset_name, results in all_results.items():
        acc = results.get("accuracy", {}).get("overall", 0.0)
        print(f"{dataset_name}: {acc:.4f}")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
