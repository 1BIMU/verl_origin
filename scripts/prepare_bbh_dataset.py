import os
import argparse
import datasets


BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

INSTRUCTION = "Please think step by step and give your answer. Put your final answer after 'The answer is '."


def make_map_fn(task_name):
    def process_fn(example, idx):
        question = example["input"]
        answer = example["target"]
        return {
            "data_source": f"bbh_{task_name}",
            "prompt": [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}],
            "ability": "reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": idx, "task": task_name, "question": question, "answer": answer},
        }
    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/bbh")
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    tasks = args.tasks if args.tasks else BBH_TASKS
    all_data = []

    for task in tasks:
        print(f"Processing task: {task}")
        dataset_path = args.local_dataset_path if args.local_dataset_path else "lukaemon/bbh"
        try:
            dataset = datasets.load_dataset(dataset_path, task, split="test")
            dataset = dataset.map(function=make_map_fn(task), with_indices=True)
            all_data.append(dataset)
            task_save_path = os.path.join(local_save_dir, f"bbh_{task}.parquet")
            dataset.to_parquet(task_save_path)
            print(f"  Saved {len(dataset)} examples to {task_save_path}")
        except Exception as e:
            print(f"  Error processing {task}: {e}")
            continue

    if all_data:
        combined = datasets.concatenate_datasets(all_data)
        combined_path = os.path.join(local_save_dir, "bbh_all.parquet")
        combined.to_parquet(combined_path)
        print(f"\nSaved combined dataset ({len(combined)} examples) to {combined_path}")


if __name__ == "__main__":
    main()
