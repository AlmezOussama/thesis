from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from vllm import LLM, SamplingParams
import pandas as pd
import json
import os
import ast
import numpy as np
import torch
import re
from transformers import AutoTokenizer


######## Data ########
def split_dictionary(data: dict):
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


def create_data_frame(test_run: bool = True, pad_value: int = -1) -> pd.DataFrame:
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"
    challenges_path = data_dir / "arc-agi_evaluation_challenges.json"
    solutions_path = data_dir / "arc-agi_evaluation_solutions.json"

    with open(challenges_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)
        challenges, _ = split_dictionary(challenges)

    solutions = None
    if test_run:
        with open(solutions_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

    data = []
    for file_name, grids in challenges.items():
        train_grids = grids.get("train", [])
        test_inputs = grids.get("test", [])
        if not test_inputs:
            continue

        if test_run:
            parts = file_name.split("_")
            base_key = parts[0]
            test_nr = int(parts[1]) if len(parts) > 1 else 0

            test_outputs = solutions.get(base_key, [])
            true_out = test_outputs[test_nr] if test_nr < len(test_outputs) else [[pad_value]]
            test_outputs_transformed = [{"output": true_out}]

            combined_tests = [{"input": test_inputs[0]["input"], "output": true_out}]
        else:
            test_outputs_transformed = [{"output": [[pad_value]]}]
            combined_tests = test_inputs

        data.append(
            {
                "file_name": file_name,
                "train": train_grids,
                "test_input": test_inputs,
                "test_output": test_outputs_transformed,
                "test": combined_tests,
            }
        )

    return pd.DataFrame(data)


######## Two-phase generation configs ########
@dataclass
class TwoPhaseConfig:
    think_tokens: int = 4096
    think_temp: float = 1.2
    think_top_p: float = 0.95
    think_top_k: int = 20

    answer_tokens: int = 2048
    answer_temp: float = 0.2
    answer_top_p: float = 1.0


######## LLM ########
def init_llm_and_tokenizer(model_subdir: str = "qwen_8b", cuda_visible: str = "1"):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = project_root / "models" / model_subdir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=80000,
        enforce_eager=True,
    )
    return llm, tokenizer


######## Helpers ########
def extract_final_grid(text: str, fallback: Optional[List[List[int]]] = None) -> List[List[int]]:
    if fallback is None:
        fallback = [[0]]

    def is_valid_grid(grid):
        if not isinstance(grid, list) or not grid:
            return False
        if not all(isinstance(r, list) and r for r in grid):
            return False
        try:
            for row in grid:
                for v in row:
                    if v is Ellipsis:
                        return False
                    if not isinstance(v, (int, np.integer)):
                        return False
            return True
        except Exception:
            return False

    if not text:
        return fallback

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

    return fallback


def pad_array_with_value(array, target_shape, pad_value=-1):
    padded = np.full(target_shape, pad_value, dtype=int)
    if not isinstance(array, list):
        return padded
    for i, row in enumerate(array):
        if not isinstance(row, list):
            continue
        for j, val in enumerate(row):
            if i < target_shape[0] and j < target_shape[1]:
                try:
                    padded[i, j] = int(val)
                except Exception:
                    padded[i, j] = pad_value
    return padded


def grid_accuracy(generated_output, correct_output, pad_value=-1):
    if not generated_output or not correct_output:
        return False, 0.0

    try:
        gen_h = len(generated_output)
        gen_w = len(generated_output[0]) if gen_h > 0 else 0
        cor_h = len(correct_output)
        cor_w = len(correct_output[0]) if cor_h > 0 else 0
    except Exception:
        return False, 0.0

    max_rows = max(gen_h, cor_h)
    max_cols = max(gen_w, cor_w)
    target_shape = (max_rows, max_cols)

    padded_generated = pad_array_with_value(generated_output, target_shape, pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape, pad_value)

    total_pixels = max_rows * max_cols
    correct_pixels = int(np.sum(padded_generated == padded_correct))
    correct_percentage = (correct_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    is_correct = (correct_pixels == total_pixels)
    return is_correct, correct_percentage


def messages_to_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _ensure_wrapped_think(text: str) -> str:
    t = (text or "").strip()
    if "<think>" not in t:
        t = "<think>\n" + t
    if "</think>" not in t:
        t = t + "\n</think>"
    return t


def generate_two_phase_tagged_batch_from_messages(
    llm,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    start_tag: str,
    end_tag: str,
    cfg: TwoPhaseConfig,
) -> Tuple[List[str], List[str]]:
    base_prompts = [messages_to_prompt(tokenizer, msgs) for msgs in messages_list]

    think_params = SamplingParams(
        temperature=cfg.think_temp,
        top_p=cfg.think_top_p,
        top_k=cfg.think_top_k,
        max_tokens=cfg.think_tokens,
        stop=[start_tag],
    )
    think_outs = llm.generate(base_prompts, think_params)

    thinks: List[str] = []
    for o in think_outs:
        txt = o.outputs[0].text.strip() if o.outputs else ""
        thinks.append(_ensure_wrapped_think(txt))

    phase2_prompts = [bp + th + start_tag for bp, th in zip(base_prompts, thinks)]

    answer_params = SamplingParams(
        temperature=cfg.answer_temp,
        top_p=cfg.answer_top_p,
        max_tokens=cfg.answer_tokens,
        stop=[end_tag],
    )
    ans_outs = llm.generate(phase2_prompts, answer_params)

    payloads: List[str] = []
    for o in ans_outs:
        txt = o.outputs[0].text.strip() if o.outputs else ""
        if txt.lstrip().startswith(start_tag):
            txt = txt.split(start_tag, 1)[1].lstrip()
        payloads.append(txt)

    return thinks, payloads


######## Inference (two-phase + micro-batching) ########
def run(
    df: pd.DataFrame,
    llm,
    tokenizer,
    gen_cfg: TwoPhaseConfig,
    max_tasks: Optional[int] = None,
    total_samples: int = 102,
    batch_size: int = 6,
    pad_value: int = -1,
):
    system_prompt = (
        "You are a puzzle solving wizard. You are given a puzzle from the "
        "Abstraction and Reasoning Corpus developed by François Chollet.\n"
        "You must reason in <think> tags, then output ONLY the final grid inside <answer>...</answer>.\n"
        "Do not use ellipsis. Use explicit integers.\n"
        "No extra text after </answer>.\n"
    )

    user_template = (
        "Here are the training examples (input and output pairs):\n"
        "{training_data}\n\n"
        "Now, predict the output for this test input:\n"
        "{input_test_data}\n"
    )

    tasks_to_run = df if max_tasks is None else df.head(max_tasks)
    print(f"Running inference on {len(tasks_to_run)} tasks...")

    all_results: Dict[str, Any] = {}
    evaluation_results: Dict[str, Any] = {}

    for _, row in tasks_to_run.iterrows():
        training_data = row["train"]
        input_test_data = row["test_input"]
        true_grid = row["test_output"][0]["output"]
        file_name = row["file_name"]

        user_message = user_template.format(training_data=training_data, input_test_data=input_test_data)

        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        task_predictions = []

        num_batches = (total_samples + batch_size - 1) // batch_size

        for b in range(num_batches):
            remaining = total_samples - len(task_predictions)
            current_batch_size = min(batch_size, remaining)

            print(f"Generating batch {b+1}/{num_batches} ({current_batch_size} samples) for {file_name}...")

            messages_list = [base_messages for _ in range(current_batch_size)]

            _, payloads = generate_two_phase_tagged_batch_from_messages(
                llm=llm,
                tokenizer=tokenizer,
                messages_list=messages_list,
                start_tag="<answer>",
                end_tag="</answer>",
                cfg=gen_cfg,
            )

            for p in payloads:
                raw = p.strip()
                grid = extract_final_grid(raw, fallback=[[pad_value]])
                task_predictions.append({"raw_output": raw, "grid": grid})

        all_results[file_name] = task_predictions
        evaluation_results[file_name] = []

        for pred in task_predictions:
            is_match, acc = grid_accuracy(pred["grid"], true_grid, pad_value=pad_value)
            evaluation_results[file_name].append(
                {
                    "test_input": input_test_data,
                    "ground_truth": true_grid,
                    "predicted_grid": pred["grid"],
                    "raw_output": pred["raw_output"],
                    "match": is_match,
                    "cell_accuracy": acc,
                }
            )

        print(f"\n=== Task: {file_name} ===")
        print(f"Ground truth:\n{true_grid}\n")
        for i, res in enumerate(evaluation_results[file_name]):
            print(f"[Sample {i+1}] Match={res['match']}, Accuracy={res['cell_accuracy']:.2f}")
            print("--- Raw output (answer payload) ---")
            print(res["raw_output"])
            print("--- Extracted grid ---")
            print(res["predicted_grid"])
            print("------")

    return all_results, evaluation_results


######## Main ########
def main():
    pad_value = -1
    df = create_data_frame(test_run=True, pad_value=pad_value)
    df = df.iloc[[0, 188, 5, 91, 136, 376, 188, 5, 91, 136, 376, 10, 11, 47, 78, 169, 52, 163, 79, 238, 336, 93, 399, 210, 263, 292, 138, 316, 118, 257, 388, 394, 357, 354, 275, 233, 276, 201, 385, 383, 43, 189, 3, 303, 309
]]
    #
    MODEL_SUBDIR = "qwen_8b"
    CUDA_VISIBLE = "1"

    gen_cfg = TwoPhaseConfig(
        think_tokens=6000,
        think_temp=1.2,
        think_top_p=0.95,
        think_top_k=20,
        answer_tokens=2000,
        answer_temp=0.5,
        answer_top_p=1.0,
    )

    llm, tokenizer = init_llm_and_tokenizer(model_subdir=MODEL_SUBDIR, cuda_visible=CUDA_VISIBLE)
    all_results, evaluation_results = run(
        df=df,
        llm=llm,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        max_tasks=len(df),
        total_samples=8,
        batch_size=8,
        pad_value=pad_value,
    )

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_rows = []
    for file_name, results_list in evaluation_results.items():
        for i, res in enumerate(results_list):
            csv_rows.append(
                {
                    "file_name": file_name,
                    "sample_id": i + 1,
                    "test_input": res["test_input"],
                    "ground_truth": res["ground_truth"],
                    "predicted_grid": res["predicted_grid"],
                    "match": res["match"],
                    "cell_accuracy": res["cell_accuracy"],
                }
            )

    df_csv = pd.DataFrame(csv_rows)
    csv_path = output_dir / (
        f"qwen8b_twophase_{timestamp}"
        f"_tT{gen_cfg.think_temp}_tP{gen_cfg.think_top_p}_tK{gen_cfg.think_top_k}"
        f"_aT{gen_cfg.answer_temp}_aP{gen_cfg.answer_top_p}"
        f"_k{102}.csv"
    )
    df_csv.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
