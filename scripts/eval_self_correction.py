#!/usr/bin/env python3
"""
端到端的 self-correction 评测脚本
1. 调用 main_generation.py 生成模型回答
2. 调用 llm_judge_self_correction.py 进行 self-correction 评测
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


# 默认数据集路径
DEFAULT_DATASET = "data/offline_eval/math__minerva_math_2025_processed.parquet"


def run_generation(
    model_path: str,
    data_path: str,
    output_path: str,
    n_samples: int,
    n_gpus: int,
    batch_size: int,
    temperature: float,
    prompt_length: int,
    response_length: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int,
    max_model_len: int,
    tensor_parallel_size: int,
):
    """调用 main_generation.py 生成模型回答"""
    cmd = [
        sys.executable, "-m", "verl.trainer.main_generation",
        f"model.path={model_path}",
        f"data.path={data_path}",
        f"data.output_path={output_path}",
        f"data.n_samples={n_samples}",
        f"data.batch_size={batch_size}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"rollout.temperature={temperature}",
        f"rollout.prompt_length={prompt_length}",
        f"rollout.response_length={response_length}",
        f"rollout.gpu_memory_utilization={gpu_memory_utilization}",
        f"rollout.max_num_batched_tokens={max_num_batched_tokens}",
        f"rollout.max_model_len={max_model_len}",
        f"rollout.tensor_model_parallel_size={tensor_parallel_size}",
        f"rollout.do_sample=True",
        f"rollout.enforce_eager=True",
    ]
    print(f"Running generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_self_correction_eval(
    input_path: str,
    output_path: str,
    judge_model: str,
    max_workers: int,
):
    """调用 llm_judge_self_correction.py 进行评测"""
    cmd = [
        sys.executable, "scripts/llm_judge_self_correction.py",
        f"--input_path={input_path}",
        f"--output_path={output_path}",
        f"--model={judge_model}",
        f"--max_workers={max_workers}",
    ]
    print(f"Running self-correction evaluation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="端到端 self-correction 评测")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True, help="模型路径列表")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="评测数据集路径")
    parser.add_argument("--output_dir", type=str, default="./output/self_correction_eval", help="输出目录")

    # 生成参数 (与训练对齐)
    parser.add_argument("--n_samples", type=int, default=1, help="每个 prompt 采样次数")
    parser.add_argument("--n_gpus", type=int, default=4, help="GPU 数量")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
    parser.add_argument("--prompt_length", type=int, default=512, help="最大 prompt 长度")
    parser.add_argument("--response_length", type=int, default=8192, help="最大 response 长度")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU 显存利用率")
    parser.add_argument("--max_num_batched_tokens", type=int, default=49152, help="最大 batch token 数")
    parser.add_argument("--max_model_len", type=int, default=12288, help="最大模型长度")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor parallel size")

    # LLM Judge 参数
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Judge 模型")
    parser.add_argument("--max_workers", type=int, default=32, help="并发 worker 数")

    # 控制参数
    parser.add_argument("--skip_generation", action="store_true", help="跳过生成步骤")
    parser.add_argument("--skip_judge", action="store_true", help="跳过评测步骤")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Self-Correction Evaluation Pipeline")
    print("=" * 60)
    print(f"Models: {args.model_paths}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"n_samples: {args.n_samples}, temperature: {args.temperature}")
    print(f"max_num_batched_tokens: {args.max_num_batched_tokens}")
    print("=" * 60)

    all_results = {}

    for model_path in args.model_paths:
        model_name = os.path.basename(model_path.rstrip("/"))
        print(f"\n{'=' * 50}")
        print(f"Processing: {model_name}")
        print(f"{'=' * 50}")

        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        generation_output = os.path.join(model_output_dir, "generation.parquet")
        judge_output = os.path.join(model_output_dir, "self_correction_analysis.json")

        # Step 1: Generation
        if not args.skip_generation:
            print(f"\n--- Step 1: Generation ---")
            run_generation(
                model_path=model_path,
                data_path=args.dataset,
                output_path=generation_output,
                n_samples=args.n_samples,
                n_gpus=args.n_gpus,
                batch_size=args.batch_size,
                temperature=args.temperature,
                prompt_length=args.prompt_length,
                response_length=args.response_length,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        else:
            print(f"\n--- Step 1: Generation (skipped) ---")

        # Step 2: Self-Correction Evaluation
        if not args.skip_judge:
            print(f"\n--- Step 2: Self-Correction Evaluation ---")
            if os.path.exists(generation_output):
                run_self_correction_eval(
                    input_path=generation_output,
                    output_path=judge_output,
                    judge_model=args.judge_model,
                    max_workers=args.max_workers,
                )
                all_results[model_name] = judge_output
            else:
                print(f"Warning: {generation_output} not found, skipping evaluation")
        else:
            print(f"\n--- Step 2: Self-Correction Evaluation (skipped) ---")

    print(f"\n{'=' * 60}")
    print("=== Pipeline Complete ===")
    print(f"{'=' * 60}")
    for model_name, result_path in all_results.items():
        print(f"{model_name}: {result_path}")


if __name__ == "__main__":
    main()
