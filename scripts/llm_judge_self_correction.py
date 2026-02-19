import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You are an expert at analyzing reasoning traces. Your task is to identify and count self-correction instances in a model's response.

Self-correction is when the model:
1. Realizes it made a mistake and corrects it
2. Reconsiders its approach mid-reasoning
3. Backtracks from an incorrect path
4. Uses phrases like "wait", "actually", "let me reconsider", "I made an error", "that's not right", etc.

For each response, you should:
1. Count the number of distinct self-correction instances
2. Briefly describe each self-correction (what was corrected)
3. Assess the quality of self-corrections (did they lead to correct answers?)

Output your analysis in the following JSON format:
{
    "self_correction_count": <integer>,
    "corrections": [
        {
            "description": "<brief description of what was corrected>",
            "trigger_phrase": "<the phrase that triggered the correction>",
            "successful": <true/false, whether the correction improved the answer>
        }
    ],
    "overall_assessment": "<brief assessment of the model's self-correction ability>"
}
"""

USER_PROMPT_TEMPLATE = """Analyze the following model response for self-correction behavior:

Question: {question}

Model Response:
{response}

Ground Truth Answer: {ground_truth}

Please identify all self-correction instances and provide your analysis in JSON format."""


def analyze_response(client, model, question, response, ground_truth, max_retries=3):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        ground_truth=ground_truth,
    )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"self_correction_count": -1, "corrections": [], "overall_assessment": "Failed to parse", "error": "JSON decode error"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"self_correction_count": -1, "corrections": [], "overall_assessment": f"Error: {str(e)}", "error": str(e)}

    return {"self_correction_count": -1, "corrections": [], "overall_assessment": "Max retries exceeded", "error": "Max retries exceeded"}


def process_dataset(input_path, output_path, model="gpt-4o", max_workers=10, sample_size=None, api_base=None, api_key=None):
    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )
    df = pd.read_parquet(input_path)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    results = []
    total_corrections = 0
    successful_corrections = 0

    tasks = []
    for idx, row in df.iterrows():
        question = row.get("extra_info", {}).get("question", "")
        if not question and "prompt" in row:
            prompt = row["prompt"]
            # prompt 可能是 list 或 numpy.ndarray
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            if isinstance(prompt, list) and len(prompt) > 0:
                question = prompt[0].get("content", "")

        responses = row.get("responses", [])
        # responses 可能是 None 或 numpy.ndarray
        if responses is None:
            responses = []
        if hasattr(responses, "tolist"):
            responses = responses.tolist()
        ground_truth = row.get("reward_model", {}).get("ground_truth", "")

        for resp_idx, response in enumerate(responses):
            tasks.append({
                "idx": idx,
                "resp_idx": resp_idx,
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "data_source": row.get("data_source", "unknown"),
            })

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(analyze_response, client, model, task["question"], task["response"], task["ground_truth"]): task
            for task in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Analyzing"):
            task = future_to_task[future]
            try:
                analysis = future.result()
                result = {
                    "idx": task["idx"],
                    "resp_idx": task["resp_idx"],
                    "data_source": task["data_source"],
                    "question": task["question"][:200],
                    "response": task["response"][:500],
                    "ground_truth": task["ground_truth"],
                    **analysis,
                }
                results.append(result)

                if analysis.get("self_correction_count", 0) > 0:
                    total_corrections += analysis["self_correction_count"]
                    for correction in analysis.get("corrections", []):
                        if correction.get("successful", False):
                            successful_corrections += 1
            except Exception as e:
                print(f"Error processing task {task['idx']}: {e}")

    summary = {
        "total_samples": len(df),
        "total_responses_analyzed": len(tasks),
        "total_self_corrections": total_corrections,
        "successful_corrections": successful_corrections,
        "avg_corrections_per_response": total_corrections / len(tasks) if tasks else 0,
        "correction_success_rate": successful_corrections / total_corrections if total_corrections > 0 else 0,
    }

    by_source = {}
    for result in results:
        source = result["data_source"]
        if source not in by_source:
            by_source[source] = {"count": 0, "total_corrections": 0, "successful_corrections": 0, "errors": 0}
        by_source[source]["count"] += 1
        # 只统计有效的 self_correction_count (>= 0)
        count = result.get("self_correction_count", 0)
        if count >= 0:
            by_source[source]["total_corrections"] += count
        else:
            by_source[source]["errors"] += 1
        for correction in result.get("corrections", []):
            if correction.get("successful", False):
                by_source[source]["successful_corrections"] += 1

    summary["by_data_source"] = by_source

    output = {"summary": summary, "detailed_results": results}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Self-Correction Analysis Summary ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total responses analyzed: {summary['total_responses_analyzed']}")
    print(f"Total self-corrections found: {summary['total_self_corrections']}")
    print(f"Successful corrections: {summary['successful_corrections']}")
    print(f"Avg corrections per response: {summary['avg_corrections_per_response']:.2f}")
    print(f"Correction success rate: {summary['correction_success_rate']:.2%}")
    print(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./output/self_correction_analysis.json")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--api_base", type=str, default='https://api.ai-gaochao.cn/v1', help="API base URL (e.g., https://api.openai.com/v1)")
    parser.add_argument("--api_key", type=str, default='sk-zaCd8xg5am8ioEPiA1CdD1946dF54e8e9c93428a34EaEc5b', help="API key (默认使用 OPENAI_API_KEY 环境变量)")
    args = parser.parse_args()

    process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        max_workers=args.max_workers,
        sample_size=args.sample_size,
        api_base=args.api_base,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
