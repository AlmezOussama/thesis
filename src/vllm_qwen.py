from vllm import LLM, SamplingParams, LLMEngine
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import pandas as pd
import json
import os
import ast
import re
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from pathlib import Path
from datetime import datetime


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
                result[new_key] = {
                    "test": [test_item],
                    "train": train_list
                }
                split_files.append(new_key)
        else:
            result[key] = value
    return result, split_files

def create_data_frame(test_run=True):
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"
    challenges = data_dir / "arc-agi_evaluation_challenges.json"
    solutions = data_dir / "arc-agi_evaluation_solutions.json"

    if test_run:
        with open(challenges) as f:
            challenges = json.load(f)
            challenges, split_files = split_dictionary(challenges) 

        with open(solutions) as f:
            solutions = json.load(f)

    data = []
            
    for file_name, grids in challenges.items():
        train_grids = grids.get('train', [])
        test_inputs = grids.get('test', [])
        
        if test_run:
            parts = file_name.split('_')
            if len(parts) > 1:
                test_nr = int(parts[1])
            else:
                test_nr = 0
            
            test_outputs = solutions.get(parts[0], [])
            test_outputs_transformed = [{'output': test_outputs[test_nr]}]
            combined_tests = [{'input': test_inputs[0]['input'], 'output': test_outputs_transformed[0]['output']}]
        
        data.append({
                'file_name': file_name,
                'train': train_grids,
                'test_input': test_inputs,
                'test_output': test_outputs_transformed if test_run else [[0, 0]],
                'test': combined_tests if test_run else test_inputs
        })

    df = pd.DataFrame(data)
    return df

######## LLM ########
def init_llm(model_subdir="qwen4b_awq"):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = project_root / "models" / model_subdir

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        n=4,
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
    )

    return llm, sampling_params

import ast
import re

######## Extract Grid ########
def extract_output_grid(text):
    """
    Extracts the first valid output grid from LLM text.
    Returns a Python list (grid), or None if extraction fails.
    """
    try:
        # Match first [[...]] array in the text
        match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
        if match:
            grid_str = match.group(0)
            grid = ast.literal_eval(grid_str)
            return grid
        else:
            return None
    except Exception as e:
        print("Failed to extract grid:", e)
        return None

######## Inference ########
def run(df, llm, sampling_params, max_tasks=None):

    system_prompt = (
        "You are a puzzle solving wizard. You are given a puzzle from the "
        "Abstraction and Reasoning Corpus developed by François Chollet. "
        "To solve these puzzles, focus on the hidden rule that transforms a "
        "grid of numbers into a new grid of numbers. Try to infer the rule "
        "from the given examples!"
    )

    user_message_template = (
        "Here are the example input and output pairs from which you should learn "
        "the underlying rule to later predict the output for the given test input:\n"
        "----------------------------------------\n"
        "{training_data}\n"
        "----------------------------------------\n"
        "Now, solve the following puzzle based on its input grid by applying the "
        "rules you have learned from the training data:\n"
        "----------------------------------------\n"
        "[{{'input': {input_test_data}, 'output': [[]]}}]\n"
        "----------------------------------------\n"
        "What is the output grid? Provide ONLY the final output grid in the shown format."
    )

    # Determine tasks to run
    tasks_to_run = df if max_tasks is None else df.head(max_tasks)
    print(f"Running inference on {len(tasks_to_run)} tasks...")

    # Collection to store all results
    all_task_results = {}

    for _, row in tasks_to_run.iterrows():
        # Fill in the user template
        training_data = row["train"]
        input_test_data = row["test_input"]

        user_message = user_message_template.format(
            training_data=training_data,
            input_test_data=input_test_data
        )

        # Combine system + user prompt into a single string
        prompt = f"{system_prompt}\n\n{user_message}"

        # Generate outputs
        outputs = llm.generate([prompt], sampling_params)

        # Extract grids for each sample
        task_samples = []
        for j, sample in enumerate(outputs[0].outputs):
            grid = extract_output_grid(sample.text)
            if grid is not None:
                task_samples.append(grid)
            else:
                print(f"[Warning] Sample {j+1} did not produce a valid grid.")

        # Store all sampled grids for this task
        all_task_results[row['file_name']] = task_samples

        # Print task summary
        print(f"\n=== Task: {row['file_name']} ===")
        for idx, grid in enumerate(task_samples):
            print(f"[Sample {idx+1}]")
            for row_line in grid:
                print(row_line)
            print()

    return all_task_results


def main():
    df = create_data_frame(test_run=True)
    llm, sampling_params = init_llm()
    run(df, llm, sampling_params, max_tasks=2)

    del llm  
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

