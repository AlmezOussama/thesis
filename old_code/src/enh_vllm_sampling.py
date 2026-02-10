from vllm import LLM, SamplingParams
from pathlib import Path
import pandas as pd
import json
import os
import ast
import numpy as np
import torch
from datetime import datetime
import re
import time
from transformers import AutoTokenizer


######## Data ########
def split_dictionary(data):
    result = {}
    split_files = []
    for key, value in data.items():
        test_list = value.get("test", [])
        train_list = value.get("train", [])
        if len(test_list) > 1:
            for idx, test_item in enumerate(test_list):
                new_key = f"{key}_{idx}"
                result[new_key] = {"test": [test_item], "train": train_list}
                split_files.append(new_key)
        else:
            result[key] = value
    return result, split_files


def create_data_frame(test_run=True):
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"
    challenges_path = data_dir / "arc-agi_evaluation_challenges.json"
    solutions_path = data_dir / "arc-agi_evaluation_solutions.json"

    if test_run:
        with open(challenges_path) as f:
            challenges = json.load(f)
            challenges, split_files = split_dictionary(challenges)
        with open(solutions_path) as f:
            solutions = json.load(f)

    data = []
    for file_name, grids in challenges.items():
        train_grids = grids.get('train', [])
        test_inputs = grids.get('test', [])

        if test_run:
            parts = file_name.split('_')
            test_nr = int(parts[1]) if len(parts) > 1 else 0
            test_outputs = solutions.get(parts[0], [])
            test_outputs_transformed = [{'output': test_outputs[test_nr]}]
            combined_tests = [{'input': test_inputs[0]['input'],
                               'output': test_outputs_transformed[0]['output']}]

        data.append({
            'file_name': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs_transformed if test_run else [[0, 0]],
            'test': combined_tests if test_run else test_inputs
        })

    return pd.DataFrame(data)


######## LLM ########
def init_llm(model_subdir="q3_think_f8"):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = project_root / "models" / model_subdir

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=100000,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.9,
        top_p=0.95,
        max_tokens=8192,
    )

    return llm, sampling_params



######## Helpers ########
def extract_final_grid(text):
    def is_valid_grid(grid):
        if not isinstance(grid, list) or not all(isinstance(r, list) for r in grid):
            return False
        try:
            for row in grid:
                for v in row:
                    if not isinstance(v, (int, float)):
                        return False
            return True
        except Exception:
            return False

    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        try:
            grid_candidate = ast.literal_eval(content)
            if is_valid_grid(grid_candidate):
                return grid_candidate
        except Exception:
            pass  

    grid_matches = re.findall(r"\[\[.*?\]\]", text, re.DOTALL)
    for gm in grid_matches[::-1]:
        try:
            grid_candidate = ast.literal_eval(gm)
            if is_valid_grid(grid_candidate):
                return grid_candidate
        except Exception:
            continue

    return [[0]]


def pad_array_with_value(array, target_shape, pad_value=0):
    padded = np.full(target_shape, pad_value, dtype=int)
    for i, row in enumerate(array):
        for j, val in enumerate(row):
            if i < target_shape[0] and j < target_shape[1]:
                padded[i, j] = val
    return padded


def grid_accuracy(generated_output, correct_output, pad_value=0):
    if not generated_output or not correct_output:
        return False, 0.0

    max_rows = max(len(generated_output), len(correct_output))
    max_cols = max(len(generated_output[0]), len(correct_output[0]))
    target_shape = (max_rows, max_cols)

    padded_generated = pad_array_with_value(generated_output, target_shape, pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape, pad_value)

    total_pixels = max_rows * max_cols
    correct_pixels = np.sum(padded_generated == padded_correct)
    correct_percentage = (correct_pixels / total_pixels) * 100

    is_correct = (correct_pixels == total_pixels)
    return is_correct, correct_percentage


######## Inference ########
def run(df, llm, sampling_params, max_tasks=None, total_samples=100, batch_size=10):
    system_prompt = (
        "You are a puzzle solving wizard. You are given a puzzle from the "
        "Abstraction and Reasoning Corpus developed by François Chollet. "
        "To solve these puzzles, think carefully and reason step by step "
        "inside <think> tags, and provide your final grid inside <answer> tags."
    )

    user_template = (
        "Here are the training examples (input and output pairs):\n"
        "{training_data}\n\n"
        "Now, predict the output for this test input:\n"
        "{input_test_data}\n\n"
        "Please reason in <think> tags and give the final grid in <answer> tags."
    )

    tasks_to_run = df if max_tasks is None else df.head(max_tasks)
    print(f"Running inference on {len(tasks_to_run)} tasks...")

    all_results, evaluation_results = {}, {}

    for _, row in tasks_to_run.iterrows():
        training_data = row["train"]
        input_test_data = row["test_input"]
        true_grid = row["test_output"][0]["output"]
        file_name = row["file_name"]

        user_message = user_template.format(
            training_data=training_data,
            input_test_data=input_test_data
        )

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        task_predictions = []

        # Compute how many batches we need
        num_batches = (total_samples + batch_size - 1) // batch_size

        for b in range(num_batches):
            current_batch_size = min(batch_size, total_samples - len(task_predictions))
            batch_params = SamplingParams(
                n=current_batch_size,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_tokens=sampling_params.max_tokens,
            )

            print(f"Generating batch {b+1}/{num_batches} ({current_batch_size} samples) for {file_name}...")
            outputs = llm.generate([prompt], batch_params)

            for sample in outputs[0].outputs:
                text = sample.text.strip()
                grid = extract_final_grid(text)
                task_predictions.append({
                    "raw_output": text,
                    "grid": grid
                })

        all_results[file_name] = task_predictions
        evaluation_results[file_name] = []

        for pred in task_predictions:
            is_match, acc = grid_accuracy(pred["grid"], true_grid)
            evaluation_results[file_name].append({
                "test_input": input_test_data,
                "ground_truth": true_grid,
                "predicted_grid": pred["grid"],
                "raw_output": pred["raw_output"],
                "match": is_match,
                "cell_accuracy": acc
            })

        # Print summary
        print(f"\n=== Task: {file_name} ===")
        print(f"Ground truth:\n{true_grid}\n")
        for i, res in enumerate(evaluation_results[file_name]):
            print(f"[Sample {i+1}] Match={res['match']}, Accuracy={res['cell_accuracy']:.2f}")
            print("--- Raw output ---")
            print(res["raw_output"])
            print("--- Extracted grid ---")
            print(res["predicted_grid"])
            print("------")

    return all_results, evaluation_results



######## Main ########
def main():
    df = create_data_frame(test_run=True)
    df = df.iloc[[0, 11, 169, 52, 163, 79, 238, 336, 93, 399, 138, 316, 118, 257, 388, 394, 201, 385, 43, 189, 3]]
    
    llm, sampling_params = init_llm()
    all_results, evaluation_results = run(df, llm, sampling_params, max_tasks=len(df))

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_rows = []
    for file_name, results_list in evaluation_results.items():
        for i, res in enumerate(results_list):
            csv_rows.append({
                "file_name": file_name,
                "sample_id": i + 1,
                "test_input": res["test_input"],
                "ground_truth": res["ground_truth"],
                "predicted_grid": res["predicted_grid"],
                "match": res["match"],
                "cell_accuracy": res["cell_accuracy"],
                "raw_output": res["raw_output"]
            })

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to {csv_path}")

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
