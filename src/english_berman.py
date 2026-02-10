"""
ARC English-Only Berman-Style Pipeline (two-phase thinking, micro-batched generation)

Keeps:
- two-phase generation (think -> answer) via generate_two_phase_tagged_batch_from_messages
- micro-batching for n_init and n_individual_revisions
- English instruction evolution (init + individual revisions + pooling)
- English executor for train fitness + test prediction
- logging to file (TeeLogger)
- padding with -1 and empty grid [[-1]]
- minimal CSV output: file_name, final_solution, match, accuracy

Removes:
- entire Python generation/selection/revision/execution pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import os
import re
import ast
import json
import builtins

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
@dataclass
class GenPhaseConfig:
    think_tokens: int
    think_temp: float
    think_top_p: float
    think_top_k: int
    answer_tokens: int
    answer_temp: float
    answer_top_p: float


@dataclass
class SearchConfig:
    n_init: int
    top_k: int
    n_individual_revisions: int
    n_pool_revisions: int


@dataclass
class BatchConfig:
    mini_batch_init: int
    mini_batch_revisions: int


@dataclass
class RunConfig:
    instruction: GenPhaseConfig
    executor: GenPhaseConfig
    pool: GenPhaseConfig
    search: SearchConfig
    batch: BatchConfig
    pad_value: int = -1
    empty_grid: List[List[int]] = None

    def __post_init__(self):
        if self.empty_grid is None:
            self.empty_grid = [[self.pad_value]]


# =========================
# Logging
# =========================
class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.log_path, "w", encoding="utf-8")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def print(self, *args, **kwargs):
        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        msg = sep.join(str(a) for a in args)
        self.f.write(msg + end)
        self.f.flush()
        builtins.print(msg, end=end)


# =========================
# Utilities
# =========================
def chunk_list(xs, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


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


def create_data_frame(test_run=True, pad_value: int = -1):
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent
    data_dir = project_dir / "data"
    challenges_path = data_dir / "arc-agi_evaluation_challenges.json"
    solutions_path = data_dir / "arc-agi_evaluation_solutions.json"

    with open(challenges_path, "r") as f:
        challenges = json.load(f)
        challenges, _ = split_dictionary(challenges)

    solutions = None
    if test_run:
        with open(solutions_path, "r") as f:
            solutions = json.load(f)

    rows = []
    for file_name, grids in challenges.items():
        train_grids = grids.get("train", [])
        test_items = grids.get("test", [])
        if not test_items:
            continue

        if test_run:
            parts = file_name.split("_")
            base_key = parts[0]
            test_idx = int(parts[1]) if len(parts) > 1 else 0
            sol_list = solutions.get(base_key, [])
            true_out = sol_list[test_idx] if test_idx < len(sol_list) else [[pad_value]]
            test_output = [{"output": true_out}]
        else:
            test_output = [{"output": [[pad_value]]}]

        rows.append(
            {
                "file_name": file_name,
                "train": train_grids,
                "test_input": test_items,
                "test_output": test_output,
            }
        )

    return pd.DataFrame(rows)


def init_llm_and_tokenizer(model_subdir="q3_think_f8", cuda_visible="0"):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = project_root / "models" / model_subdir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    builtins.print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        builtins.print("Device:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=80000,
        enforce_eager=True,
    )
    return llm, tokenizer


# =========================
# Grid + scoring
# =========================
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
                    if val is Ellipsis:
                        padded[i, j] = pad_value
                    else:
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
    except Exception:
        return False, 0.0

    try:
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
    acc = (correct_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    is_correct = (correct_pixels == total_pixels)
    return is_correct, acc


def extract_grid_anywhere(text: str, empty_grid: List[List[int]]):
    def is_valid_grid(g):
        if not isinstance(g, list) or not g:
            return False
        if not all(isinstance(r, list) and r for r in g):
            return False
        for r in g:
            for v in r:
                if v is Ellipsis:
                    return False
                if not isinstance(v, (int, np.integer)):
                    return False
        return True

    if not text:
        return empty_grid

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        try:
            g = ast.literal_eval(content)
            if is_valid_grid(g):
                return g
        except Exception:
            pass

    grid_matches = re.findall(r"\[\[.*?\]\]", text, re.DOTALL)
    for gm in grid_matches[::-1]:
        try:
            g = ast.literal_eval(gm)
            if is_valid_grid(g):
                return g
        except Exception:
            continue

    return empty_grid


def extract_instruction_anywhere(text: str):
    if not text:
        return ""
    m = re.search(r"<instruction>(.*?)</instruction>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


# =========================
# Prompt building
# =========================
def messages_to_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _ensure_wrapped_think(text: str) -> str:
    t = text.strip()
    if "<think>" not in t:
        t = "<think>\n" + t
    if "</think>" not in t:
        t = t + "\n</think>"
    return t


def build_instruction_messages(train_examples) -> List[Dict[str, str]]:
    system = (
        "You solve ARC tasks by inferring deterministic transformations.\n"
        "You must think, then output an <instruction> block.\n"
        "In <instruction>, write one deterministic procedure. No hedging. Be explicit.\n"
        "Do not output code. Do not output the test answer grid.\n"
        "Output format:\n"
        "<think>...</think>\n"
        "<instruction>...</instruction>\n"
    )
    user = (
        "Training examples:\n"
        f"{train_examples}\n\n"
        "Produce the best deterministic instruction."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_revision_messages(instruction: str, train_examples, traces) -> List[Dict[str, str]]:
    system = (
        "You improve an ARC instruction.\n"
        "You must think, then output an <instruction> block.\n"
        "Fix the failures shown. Keep it deterministic. No code.\n"
        "Output format:\n"
        "<think>...</think>\n"
        "<instruction>...</instruction>\n"
    )

    fb = []
    for i, (ex, tr) in enumerate(zip(train_examples, traces)):
        fb.append(
            f"Example {i}:\n"
            f"Input: {ex['input']}\n"
            f"Predicted: {tr['pred_grid']}\n"
            f"Correct: {ex['output']}\n"
            f"Match: {tr['match']}, CellAcc: {tr['cell_accuracy']:.2f}\n"
        )
    user = (
        "Current instruction:\n"
        f"{instruction}\n\n"
        "Training feedback:\n"
        + "\n".join(fb)
        + "\n\nWrite an improved instruction."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_pool_messages(candidates: List[str], train_examples) -> List[Dict[str, str]]:
    system = (
        "You synthesize one best ARC instruction from multiple candidates.\n"
        "You must think, then output an <instruction> block.\n"
        "Combine correct parts and remove contradictions. Deterministic. No code.\n"
        "Output format:\n"
        "<think>...</think>\n"
        "<instruction>...</instruction>\n"
    )
    joined = "\n\n---\n\n".join([f"Candidate {i}:\n{c}" for i, c in enumerate(candidates)])
    user = (
        "Training examples:\n"
        f"{train_examples}\n\n"
        "Candidate instructions:\n"
        f"{joined}\n\n"
        "Produce the best single instruction."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_executor_messages(instruction: str, input_grid) -> List[Dict[str, str]]:
    system = (
        "You are an ARC grid executor.\n"
        "You must think, then output only the final grid in <answer>.\n"
        "Follow the instruction exactly.\n"
        "Output only: <answer>[[...]]</answer>\n"
        "Never use ellipsis. Write full explicit integers.\n"
        "No extra text after </answer>.\n"
        "Output format:\n"
        "<think>...</think>\n"
        "<answer>[[...]]</answer>\n"
    )
    user = (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Input grid:\n"
        f"{input_grid}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# =========================
# Two-phase generation
# =========================
def generate_two_phase_tagged_batch_from_messages(
    llm,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    start_tag: str,
    end_tag: str,
    think_cfg: GenPhaseConfig,
    answer_cfg: GenPhaseConfig,
) -> Tuple[List[str], List[str]]:
    from vllm import SamplingParams

    base_prompts = [messages_to_prompt(tokenizer, msgs) for msgs in messages_list]

    think_params = SamplingParams(
        temperature=think_cfg.think_temp,
        top_p=think_cfg.think_top_p,
        top_k=think_cfg.think_top_k,
        max_tokens=think_cfg.think_tokens,
        stop=[start_tag],
    )
    think_outs = llm.generate(base_prompts, think_params)

    thinks = []
    for o in think_outs:
        txt = o.outputs[0].text.strip() if o.outputs else ""
        thinks.append(_ensure_wrapped_think(txt))

    phase2_prompts = [bp + th + start_tag for bp, th in zip(base_prompts, thinks)]

    answer_params = SamplingParams(
        temperature=answer_cfg.answer_temp,
        top_p=answer_cfg.answer_top_p,
        max_tokens=answer_cfg.answer_tokens,
        stop=[end_tag],
    )
    ans_outs = llm.generate(phase2_prompts, answer_params)

    payloads = []
    for o in ans_outs:
        txt = o.outputs[0].text.strip() if o.outputs else ""
        if txt.lstrip().startswith(start_tag):
            txt = txt.split(start_tag, 1)[1].lstrip()
        payloads.append(txt)

    return thinks, payloads


# =========================
# English evolution
# =========================
def fitness_for_instruction(
    llm,
    tokenizer,
    instruction: str,
    train_examples,
    cfg: RunConfig,
) -> Tuple[Tuple[int, float], List[Dict[str, Any]]]:
    exact = 0
    cell_sum = 0.0
    traces = []

    for ex in train_examples:
        msgs = build_executor_messages(instruction, ex["input"])
        _, ans_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=[msgs],
            start_tag="<answer>",
            end_tag="</answer>",
            think_cfg=cfg.executor,
            answer_cfg=cfg.executor,
        )
        pred_grid = extract_grid_anywhere(ans_payloads[0] if ans_payloads else "", cfg.empty_grid)
        is_match, acc = grid_accuracy(pred_grid, ex["output"], pad_value=cfg.pad_value)
        exact += int(is_match)
        cell_sum += acc
        traces.append(
            {
                "pred_grid": pred_grid,
                "match": is_match,
                "cell_accuracy": acc,
                "raw_answer": ans_payloads[0] if ans_payloads else "",
            }
        )

    return (exact, cell_sum), traces


def generate_instruction_candidates_microbatched(
    llm,
    tokenizer,
    train_examples,
    n_candidates: int,
    cfg: RunConfig,
) -> List[str]:
    base_msgs = build_instruction_messages(train_examples)
    msgs_list = [base_msgs for _ in range(n_candidates)]

    all_payloads = []
    for chunk in chunk_list(msgs_list, cfg.batch.mini_batch_init):
        _, instr_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=chunk,
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_cfg=cfg.instruction,
            answer_cfg=cfg.instruction,
        )
        all_payloads.extend([extract_instruction_anywhere(p) for p in instr_payloads])

    return all_payloads


def evolve_english_instruction_for_task(
    llm,
    tokenizer,
    train_examples,
    cfg: RunConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    s = cfg.search

    instructions = generate_instruction_candidates_microbatched(
        llm=llm,
        tokenizer=tokenizer,
        train_examples=train_examples,
        n_candidates=s.n_init,
        cfg=cfg,
    )

    scored = []
    for instr in instructions:
        fit, traces = fitness_for_instruction(llm, tokenizer, instr, train_examples, cfg)
        scored.append({"instruction": instr, "fitness": fit, "traces": traces})

    scored.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    best = scored[0]

    if best["fitness"][0] == len(train_examples):
        return best, scored

    parents = scored[: s.top_k]
    n_rev = min(s.n_individual_revisions, len(parents))

    rev_msgs_list = []
    for p in parents[:n_rev]:
        rev_msgs_list.append(build_revision_messages(p["instruction"], train_examples, p["traces"]))

    revised_payloads_all = []
    for chunk in chunk_list(rev_msgs_list, cfg.batch.mini_batch_revisions):
        _, revised_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=chunk,
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_cfg=cfg.instruction,
            answer_cfg=cfg.instruction,
        )
        revised_payloads_all.extend([extract_instruction_anywhere(p) for p in revised_payloads])

    revised = []
    for instr in revised_payloads_all:
        fit, traces = fitness_for_instruction(llm, tokenizer, instr, train_examples, cfg)
        revised.append({"instruction": instr, "fitness": fit, "traces": traces})

    revised.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    if revised and revised[0]["fitness"] > best["fitness"]:
        best = revised[0]

    if best["fitness"][0] == len(train_examples):
        return best, scored + revised

    pooled = []
    top_instrs = [p["instruction"] for p in scored[: s.top_k]]
    for _ in range(s.n_pool_revisions):
        msgs = build_pool_messages(top_instrs, train_examples)
        _, payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=[msgs],
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_cfg=cfg.pool,
            answer_cfg=cfg.pool,
        )
        instr = extract_instruction_anywhere(payloads[0] if payloads else "")
        fit, traces = fitness_for_instruction(llm, tokenizer, instr, train_examples, cfg)
        pooled.append({"instruction": instr, "fitness": fit, "traces": traces})

    pooled.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    if pooled and pooled[0]["fitness"] > best["fitness"]:
        best = pooled[0]

    return best, scored + revised + pooled


# =========================
# English-only evaluation on test
# =========================
def predict_test_with_instruction(
    llm,
    tokenizer,
    instruction: str,
    test_input_grid,
    cfg: RunConfig,
) -> Tuple[List[List[int]], str]:
    msgs = build_executor_messages(instruction, test_input_grid)
    _, payloads = generate_two_phase_tagged_batch_from_messages(
        llm=llm,
        tokenizer=tokenizer,
        messages_list=[msgs],
        start_tag="<answer>",
        end_tag="</answer>",
        think_cfg=cfg.executor,
        answer_cfg=cfg.executor,
    )
    payload = payloads[0] if payloads else ""
    pred = extract_grid_anywhere(payload, cfg.empty_grid)
    return pred, payload


# =========================
# Full English-only pipeline
# =========================
def run_english_only_pipeline(
    df: pd.DataFrame,
    llm,
    tokenizer,
    logger: TeeLogger,
    cfg: RunConfig,
    max_tasks: Optional[int] = None,
):
    tasks_to_run = df if max_tasks is None else df.head(max_tasks)
    logger.print(f"Running English-only evolution pipeline on {len(tasks_to_run)} tasks...")

    out_rows = []

    for _, row in tasks_to_run.iterrows():
        file_name = row["file_name"]
        train_examples = row["train"]
        test_input_grid = row["test_input"][0]["input"]
        true_test_grid = row["test_output"][0]["output"]

        logger.print("\n" + "=" * 100)
        logger.print(f"TASK: {file_name}")
        logger.print(f"Train examples: {len(train_examples)}")

        best_eng, _ = evolve_english_instruction_for_task(
            llm=llm,
            tokenizer=tokenizer,
            train_examples=train_examples,
            cfg=cfg,
        )

        best_instruction = best_eng["instruction"]
        logger.print(f"Best English fitness (exact, cell_sum): {best_eng['fitness']}")
        logger.print("Best instruction:\n" + best_instruction[:1200] + ("..." if len(best_instruction) > 1200 else ""))

        pred_test, raw_payload = predict_test_with_instruction(
            llm=llm,
            tokenizer=tokenizer,
            instruction=best_instruction,
            test_input_grid=test_input_grid,
            cfg=cfg,
        )

        is_match, acc = grid_accuracy(pred_test, true_test_grid, pad_value=cfg.pad_value)
        logger.print(f"TEST: mode=english_only match={is_match} acc={acc:.2f}")

        out_rows.append(
            {
                "file_name": file_name,
                "final_solution": pred_test,
                "match": is_match,
                "accuracy": acc,
            }
        )

    return pd.DataFrame(out_rows)


# =========================
# Main
# =========================
def main():
    import torch

    MODEL_SUBDIR = "q3_think_f8"
    CUDA_VISIBLE = "1"

    cfg = RunConfig(
        instruction=GenPhaseConfig(
            think_tokens=4096,
            think_temp=1.2,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=1400,
            answer_temp=0.2,
            answer_top_p=1.0,
        ),
        executor=GenPhaseConfig(
            think_tokens=2048,
            think_temp=0.7,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=1400,
            answer_temp=0.2,
            answer_top_p=1.0,
        ),
        pool=GenPhaseConfig(
            think_tokens=2048,
            think_temp=0.8,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=1400,
            answer_temp=0.2,
            answer_top_p=1.0,
        ),
        search=SearchConfig(
            n_init=20,
            top_k=10,
            n_individual_revisions=10,
            n_pool_revisions=10,
        ),
        batch=BatchConfig(
            mini_batch_init=20,
            mini_batch_revisions=10,
        ),
        pad_value=-1,
    )

    df = create_data_frame(test_run=True, pad_value=cfg.pad_value)
    df = df.iloc[[0, 188, 5, 91, 136, 376, 188, 5, 91, 136, 376, 10, 11, 47, 78, 169, 52, 163, 79, 238, 336, 93, 399, 210, 263, 292, 138, 316, 118, 257, 388, 394, 357, 354, 275, 233, 276, 201, 385, 383, 43, 189, 3, 303, 309]]

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"runlog_english_only_{timestamp}.txt"
    csv_path = output_dir / f"final_english_only_{timestamp}.csv"

    logger = TeeLogger(log_path)
    logger.print(f"Logging to: {log_path}")
    logger.print("RUN CONFIG:")
    logger.print(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    llm, tokenizer = init_llm_and_tokenizer(model_subdir=MODEL_SUBDIR, cuda_visible=CUDA_VISIBLE)

    try:
        out_df = run_english_only_pipeline(
            df=df,
            llm=llm,
            tokenizer=tokenizer,
            logger=logger,
            cfg=cfg,
            max_tasks=len(df),
        )

        out_df[["file_name", "final_solution", "match", "accuracy"]].to_csv(csv_path, index=False)
        logger.print(f"\nSaved CSV to: {csv_path}")

    finally:
        try:
            del llm
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.close()


if __name__ == "__main__":
    main()
