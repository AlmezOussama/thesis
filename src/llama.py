import pandas as pd
import json
import os
import ast
import re
import numpy as np
from datasets import Dataset, concatenate_datasets
from pathlib import Path
from datetime import datetime


# For LLM
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    pipeline
)
from trl import setup_chat_format

import torch
from time import time

# Set seed
set_seed(42)


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

def init_model():


    LLAMA_3_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
    QWEN_3_CHAT_TEMPLATE = """{% set loop_messages = messages -%} {% for message in loop_messages -%} {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>' }} {% endfor -%} {% if add_generation_prompt -%} {{ '<|im_start|>assistant\n' }} {% endif -%}"""

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4",  
        bnb_4bit_compute_dtype=compute_dtype,  
    )

    model_id = "models/llama3.2_3B"

    time_start = time()
    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True, # Allow the model to use custom code from the repository
        quantization_config=bnb_config, # Apply the 4-bit or 8-bit quantization configuration
        attn_implementation='sdpa', # Use scaled-dot product attention for better performance
        torch_dtype=compute_dtype, # Set the data type for the model
        use_cache=True, # Disable caching to save memory
        device_map= {"":1}, 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE 
    #tokenizer.chat_template = QWEN_3_CHAT_TEMPLATE

    # if tokenizer.eos_token_id is None:
    #     tokenizer.eos_token = "<|im_end|>"
    
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token = tokenizer.eos_token


    time_end = time()
    print(f"Prepare model, tokenizer: {round(time_end-time_start, 3)} sec.")

    system_prompt = '''You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus developed by Francois Chollet. To solve those puzzles, focus on the hidden rule which transforms a grid of numbers into a new grid of numbers. Try to get an understanding from the examples! '''

   
    user_message_template = '''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
    ----------------------------------------
    {training_data}
    ----------------------------------------
    Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
    ----------------------------------------
    [{{'input': {input_test_data}, 'output': [[]]}}]
    ----------------------------------------
    What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:'''

    return model, tokenizer, system_prompt, user_message_template

def preprocess(task, tokenizer,system_prompt, user_message_template, test_run=True, train_mode=False):
    
    system_message = {"role": "system", "content": system_prompt}

    training_data = task['train']
    input_test_data = task['test'][0]['input']
    if test_run:
        output_test_data = task['test'][0]['output']
    else:
        output_test_data = [[0 ,0]]

    user_message_content = user_message_template.format(training_data=training_data, input_test_data=input_test_data)
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    if train_mode:
        assistant_message = {
            "role": "assistant",
            "content": str(output_test_data)
        }
        messages = [system_message, user_message, assistant_message]
    else:
        messages = [system_message, user_message]

    messages = tokenizer.apply_chat_template(messages, tokenize=False)
    if test_run:
        return {"text": messages, "solution": output_test_data, "file_name": task['file_name']}
    else:
        return {"text": messages, "file_name": task['file_name']}

def create_dataset(df, tokenizer,system_prompt, user_message_template, test_run=True):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: preprocess(task=x, tokenizer=tokenizer,system_prompt=system_prompt, user_message_template= user_message_template, test_run= test_run), batched=False, remove_columns=dataset.column_names)

    dataset = dataset.select([11, 169, 52, 163, 79, 238, 336, 93, 399, 138, 316, 118, 0, 257, 388, 394, 201, 385, 43, 189, 3, 105, 103, 188, 376, 357, 276, 383, 354, 275, 233, 68, 250, 254, 172])
    return dataset

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def filter_dataset(dataset, tokenizer):
    max_tokens = 80000
    filtered_dataset = dataset.filter(lambda x: count_tokens(x['text'], tokenizer=tokenizer) <= max_tokens)

    # Print the number of tasks filtered out and the remaining tasks
    print(f'{len(dataset)-len(filtered_dataset)} tasks contain too many tokens if we set max_tokens to {max_tokens}')
    print(f'The dataset contains {len(filtered_dataset)} tasks to evaluate the model')

    print(filtered_dataset.to_pandas().columns)
    
    return filtered_dataset

def create_pipeline(model, tokenizer):
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map= {"":1}
    )
    text_gen_pipeline.tokenizer.pad_token_id = text_gen_pipeline.model.config.eos_token_id

    # Define terminators for the pipeline
    terminators = [
        text_gen_pipeline.tokenizer.eos_token_id,
        text_gen_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    return text_gen_pipeline, terminators

# def create_pipeline(model, tokenizer):
#     text_gen_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=torch.float16,
#         device_map={"": 1}
#     )

#     if text_gen_pipeline.tokenizer.pad_token_id is None:
#         text_gen_pipeline.tokenizer.pad_token = text_gen_pipeline.eos_token or "<|im_end|>"

#     terminators = [
#         text_gen_pipeline.tokenizer.eos_token_id,
#         text_gen_pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
#     ]

#     return text_gen_pipeline, terminators

import ast

def extract_solution(text):
    if text is None:
        return [[0]]
    
    try:
        start = text.index('[[')
        end = text.index(']]', start) + 2
        array_str = text[start:end]
        
        array = ast.literal_eval(array_str)

        if not all(isinstance(row, list) for row in array):
            return [[0]]

        new_array = []
        for row in array:
            clean_row = []
            for x in row:
                try:
                    clean_row.append(int(x))  
                except (ValueError, TypeError):
                    return [[0]]  
            new_array.append(clean_row)

        return new_array

    except (ValueError, SyntaxError):
        return [[0]]


def is_rectangular(array):
    """
    Check if all rows in a 2D list have the same length.
    """
    if not array or not all(isinstance(row, list) for row in array):
        return False
    row_length = len(array[0])
    return all(len(row) == row_length for row in array)

def pad_array_with_value(array, target_shape, pad_value):
    
    padded_array = np.full(target_shape, pad_value, dtype=int)
    original_shape = np.array(array).shape
    padded_array[:original_shape[0], :original_shape[1]] = array
    return padded_array

def compare_solutions_with_padding(generated_output, correct_output, pad_value=-1):
    
    max_rows = max(len(generated_output), len(correct_output))
    max_cols = max(len(generated_output[0]), len(correct_output[0]))
    target_shape = (max_rows, max_cols)
    
    padded_generated = pad_array_with_value(generated_output, target_shape, pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape, pad_value)
    
    total_pixels = max_rows * max_cols
    correct_pixels = np.sum((padded_generated == padded_correct) & (padded_generated != pad_value) & (padded_correct != pad_value))
    correct_percentage = (correct_pixels / total_pixels) * 100
    
    is_correct = (correct_pixels == total_pixels)
    
    return is_correct, correct_percentage

def generate_solutions_passk(task, text_gen_pipeline, terminators, k=240, max_new_tokens=512, do_sample=True, temperature=0.9, top_p=0.9, start_idx=0):

    #tasks = [dict(zip(tasks.keys(), values)) for values in zip(*tasks.values())]
    prompt = task['text']
    generated_solutions = []

    
    outputs = text_gen_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=k
    )
    for output in outputs:
        generated_solution = output["generated_text"][len(prompt):]
        generated_solutions.append(generated_solution)
    
    return {f'generated_solution_{start_idx+i+1}': sol for i, sol in enumerate(generated_solutions)}

def evaluate_passk_solutions(task, test_run=True, k=240):
    
    if not test_run:
        extracted_solutions = []
        for i in range(1, k+1):
            solution_key = f'generated_solution_{i}'
            if solution_key in task:
                gen_solution = extract_solution(task[solution_key])
                extracted_solutions.append(gen_solution)
        
        return {
            'extracted_solutions': extracted_solutions,
            'pass_at_k': False,  # Cannot evaluate without ground truth
            'best_accuracy': 0.0,
            'best_solution_idx': -1
        }
    
    true_solution = task['solution']
    file_name = task['file_name']
    
    best_accuracy = 0.0
    best_solution_idx = -1
    best_solution = None
    is_any_correct = False
    
    solution_results = []
    
    # Evaluate each of the k solutions
    for i in range(1, k+1):
        solution_key = f'generated_solution_{i}'
        if solution_key in task:
            generated_text = task[solution_key]
            gen_solution = extract_solution(generated_text)

            if not is_rectangular(gen_solution) or not is_rectangular(true_solution):
                print(f"Skipping {file_name} due to jagged array.")
                continue

            # Compare with ground truth
            is_correct, correct_percentage = compare_solutions_with_padding(gen_solution, true_solution)
            
            solution_results.append({
                'solution_idx': i,
                'is_correct': is_correct,
                'accuracy': correct_percentage,
                'extracted_solution': gen_solution
            })
            
            # Track the best solution
            if correct_percentage > best_accuracy:
                best_accuracy = correct_percentage
                best_solution_idx = i
                best_solution = gen_solution
            
            # Check if any solution is completely correct
            if is_correct:
                is_any_correct = True
    
    return {
        'file_name': file_name,
        'pass_at_k': is_any_correct,
        'best_accuracy': best_accuracy,
        'best_solution_idx': best_solution_idx,
        'best_solution': best_solution,
        'k': len(solution_results),
        'all_solutions': solution_results
        
    }

def run_passk_evaluation(filtered_dataset, text_gen_pipeline, terminators,  k=240, k_run=10, test_run=True):
    print(f"Generating {k} solutions per task for pass@{k} evaluation...")
    
    
    dataset_with_solutions = Dataset.from_dict({})

    for run_idx, start_idx in enumerate(range(0,k, k_run), 1):
        print(f"Run {run_idx}/3: Generating solutions {start_idx} to {start_idx + k_run - 1}")
        
        def generation_func(task, k=k_run, text_gen_pipeline= text_gen_pipeline, terminators=terminators, start_idx=start_idx):
            return generate_solutions_passk(task, k=k_run, text_gen_pipeline= text_gen_pipeline, terminators=terminators, max_new_tokens=512, do_sample=True, temperature=0.5, top_p=0.9, start_idx=start_idx)
        
        # Generate solutions for the current run
        batch_solutions = filtered_dataset.map(
            lambda x: generation_func(x, k=k_run,text_gen_pipeline= text_gen_pipeline, terminators=terminators, start_idx=start_idx),
            batched=False
        )
        
        dataset_with_solutions = concatenate_datasets([dataset_with_solutions,batch_solutions])

    print("Evaluating solutions...")
    evaluation_results = []
    
    for i, task in enumerate(dataset_with_solutions):
        result = evaluate_passk_solutions(task, k=k)
        evaluation_results.append(result)
        
        if test_run and (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(dataset_with_solutions)} tasks")
    
    return evaluation_results, dataset_with_solutions

def analyze_passk_results(evaluation_results, k, test_run=True):
    """
    Analyze and print pass@k evaluation results.

    Parameters:
    evaluation_results (list): List of evaluation results from run_passk_evaluation.
    k (int): The k value used in pass@k evaluation.
    """
    if not test_run:
        print("Cannot analyze results - not in test mode (no ground truth available)")
        return
    
    total_tasks = len(evaluation_results)
    pass_at_k_count = sum(1 for result in evaluation_results if result['pass_at_k'])
    
    # Calculate average best accuracy
    avg_best_accuracy = sum(result['best_accuracy'] for result in evaluation_results) / total_tasks
    
    # Calculate accuracy for each attempt position
    attempt_accuracies = {}
    for i in range(1, k+1):
        accuracies = []
        for result in evaluation_results:
            for sol_result in result['all_solutions']:
                if sol_result['solution_idx'] == i:
                    accuracies.append(sol_result['accuracy'])
        if accuracies:
            attempt_accuracies[f'attempt_{i}'] = sum(accuracies) / len(accuracies)
    
    # Print results
    print(f"\n=== Pass@{k} Evaluation Results ===")
    print(f"Total tasks evaluated: {total_tasks}")
    print(f"Tasks solved with pass@{k}: {pass_at_k_count}")
    print(f"Pass@{k} success rate: {(pass_at_k_count / total_tasks) * 100:.2f}%")
    print(f"Average best accuracy: {avg_best_accuracy:.2f}%")
    
    print(f"\nAccuracy by attempt position:")
    for attempt, accuracy in attempt_accuracies.items():
        print(f"  {attempt}: {accuracy:.2f}%")
    
    # Find tasks where pass@k helped
    helped_tasks = []
    for result in evaluation_results:
        if result['pass_at_k'] and result['best_solution_idx'] > 1:
            helped_tasks.append(result)
    
    if helped_tasks:
        print(f"\nTasks where pass@{k} helped (solution wasn't the first attempt): {len(helped_tasks)}")
        print("Examples:")
        for i, task in enumerate(helped_tasks[:5]):  
            print(f"  {task['file_name']}: Best solution was attempt #{task['best_solution_idx']}")

def output_csv(evaluation_results):

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    df_eval = pd.DataFrame(evaluation_results)
    print(df_eval)
    df_eval = df_eval.iloc[:, :5]

    if not df_eval.empty:
        df_eval['best_solution'] = df_eval['best_solution'].apply(repr)
        df_eval.to_csv(f"output/{timestamp}.csv", index=False)

def run():
    k = 2
    df = create_data_frame(test_run=True)
    model, tokenizer, system_prompt, user_message_template = init_model()
    dataset = create_dataset(df=df,tokenizer=tokenizer,system_prompt=system_prompt,user_message_template=user_message_template, test_run=True)
    fdataset = filter_dataset(dataset=dataset, tokenizer=tokenizer)
    text_gen_pipeline, terminators = create_pipeline(model=model, tokenizer=tokenizer)
    evaluation_results, dataset_with_solutions = run_passk_evaluation(filtered_dataset=fdataset,k=k, text_gen_pipeline=text_gen_pipeline, terminators=terminators, k_run= 2,test_run=True)
    output_csv(evaluation_results=evaluation_results)

def main():
    run()

if __name__ == "__main__":
    main()




