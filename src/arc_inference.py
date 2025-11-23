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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
def extract_output(text):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)

    think_text = think_match.group(1).strip() if think_match else None
    answer_text = answer_match.group(1).strip() if answer_match else None

    return think_text, answer_text

######## Inference ########
def run(df, llm, sampling_params, max_tasks=None, total_samples=2, batch_size=1):
    system_prompt = (
        "You are a puzzle solving wizard. You are given a puzzle from the "
        "Abstraction and Reasoning Corpus developed by François Chollet.  "
        "Think step by step inside <think> tags. Inside <answer> tags, provide "
        "only an English explanation of how you would solve the task. Do NOT output any grid."
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
                think_text, answer_text = extract_output(text)

                task_predictions.append({
                    "raw_output": text,
                    "think": think_text,
                    "answer_text": answer_text
                })


        all_results[file_name] = task_predictions
        evaluation_results[file_name] = []

        for pred in task_predictions:
            evaluation_results[file_name].append({
                "test_input": input_test_data,
                "think": pred["think"],
                "answer_text": pred["answer_text"],
                "raw_output": pred["raw_output"],
            })


        # Print summary
        print(f"\n=== Task: {file_name} ===")
        #print(f"Ground truth:\n{true_grid}\n")
        # for i, res in enumerate(evaluation_results[file_name]):
        #     print(f"[Sample {i+1}] Match={res['match']}, Accuracy={res['cell_accuracy']:.2f}")
        #     print("--- Raw output ---")
        #     print(res["raw_output"])
        #     print("--- Extracted grid ---")
        #     print(res["predicted_grid"])
        #     print("------")

    return all_results, evaluation_results

######## Main ########
def main():
    df = create_data_frame(test_run=True)
    df = df.iloc[[0, 11]]
    #, 169, 52, 163, 79, 238, 336, 93, 399, 138, 316, 118, 257, 388, 394, 201, 385, 43, 189, 3


    llm, sampling_params = init_llm()
    all_results, evaluation_results = run(df, llm, sampling_params, max_tasks=len(df))

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_rows = []
    for file_name, results_list in evaluation_results.items():
        print(file_name)
        
        for i, res in enumerate(results_list):
            csv_rows.append({
                "file_name": file_name,
                "sample_id": i + 1,
                "test_input": res["test_input"],
                "think": res["think"],
                "answer_text": res["answer_text"],
                "raw_output": res["raw_output"]
            })



    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / f"english_results_{timestamp}.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()