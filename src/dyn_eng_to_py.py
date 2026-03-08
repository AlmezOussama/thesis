"""
ARC English→Python Evolution Pipeline (two-phase thinking, micro-batched generation)

Single RunConfig controls:
- temperatures, top_p, top_k, token limits for thinking/answer for English instruction, executor, and Python
- Berman-style search parameters
- micro-batch sizes
- python timeouts
- padding value (-1) and empty grid default [[-1]]

Update:
- Stage-wise ("funnel") revisions for English instructions:
  - revise wider early, then keep fewer candidates and revise deeper
- Stage-wise ("funnel") revisions for Python solvers:
  - revise wider early, then keep fewer candidates and revise deeper
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
import multiprocessing as mp

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
    py_n_candidates: int
    py_revision_steps: int

    # New: staged funnel revisions for English
    # Example: rev_stage_keeps=[10, 5, 2], rev_stage_rounds=[1, 2, 4]
    # Means: keep 10 and do 1 round; keep 5 and do 2 rounds; keep 2 and do 4 rounds.
    rev_stage_keeps: Optional[List[int]] = None
    rev_stage_rounds: Optional[List[int]] = None

    # New: staged funnel revisions for Python
    # Example: py_rev_stage_keeps=[10, 3, 1], py_rev_stage_rounds=[1, 2, 6]
    py_rev_stage_keeps: Optional[List[int]] = None
    py_rev_stage_rounds: Optional[List[int]] = None


@dataclass
class BatchConfig:
    mini_batch_init: int
    mini_batch_revisions: int
    mini_batch_py: int
    mini_batch_exec: int  # NEW: micro-batch size for executor fitness scoring


@dataclass
class TimeoutConfig:
    python_train_timeout_s: float
    python_test_timeout_s: float


@dataclass
class RunConfig:
    instruction: GenPhaseConfig
    executor: GenPhaseConfig
    python: GenPhaseConfig
    python_revision: GenPhaseConfig
    pool: GenPhaseConfig
    search: SearchConfig
    batch: BatchConfig
    timeout: TimeoutConfig
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


def init_llm_and_tokenizer(model_subdir="q3_think_f8"):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_dir = project_root / "models" / model_subdir

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


def build_python_messages(best_instruction: str, train_examples) -> List[Dict[str, str]]:
    system = (
        "You write correct Python ARC solvers.\n"
        "You must think, then output only Python inside <python>.\n"
        "Output exactly one function: transform(grid)\n"
        "grid is list[list[int]]; return list[list[int]]\n"
        "Deterministic, general rule. No I/O. No network.\n"
        "No imports except optional numpy as np.\n"
        "Do not use while-loops. Do not use recursion. No brute-force search.\n"
        "Complexity must be O(H*W) or O(H*W*constant).\n"
        "Output format:\n"
        "<think>...</think>\n"
        "<python>def transform(grid): ...</python>\n"
    )
    user = (
        "Best English instruction:\n"
        f"{best_instruction}\n\n"
        "Training examples:\n"
        f"{train_examples}\n\n"
        "Write transform(grid). Ensure you return the output grid."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_python_revision_messages(code: str, train_examples, failures_summary: str) -> List[Dict[str, str]]:
    system = (
        "You debug a Python ARC solver.\n"
        "You must think, then output only corrected Python inside <python>.\n"
        "Keep signature: def transform(grid)\n"
        "Fix the failures shown. Ensure you return the output grid.\n"
        "Do not use while-loops. Do not use recursion. No brute-force search.\n"
        "Complexity must be O(H*W) or O(H*W*constant).\n"
        "Output format:\n"
        "<think>...</think>\n"
        "<python>def transform(grid): ...</python>\n"
    )
    user = (
        "Current code:\n"
        f"{code}\n\n"
        "Training examples:\n"
        f"{train_examples}\n\n"
        "Failures:\n"
        f"{failures_summary}\n\n"
        "Return corrected code."
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
# Safe Python exec with timeout
# =========================
def _normalize_grid_like(out):
    if out is None:
        return None
    if not isinstance(out, list):
        return None
    if not all(isinstance(r, list) for r in out):
        return None
    return out


def _worker_exec(code: str, grid, q: mp.Queue, pad_value: int):
    try:
        safe_builtins = {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sorted": sorted,
            "int": int,
            "float": float,
            "any": any,
            "all": all,
            "map": map,
            "filter": filter,
        }
        env = {"__builtins__": safe_builtins, "np": np}
        exec(code, env, env)
        if "transform" not in env:
            q.put(("error", "no_transform_defined", [[pad_value]]))
            return
        out = env["transform"](grid)
        out = _normalize_grid_like(out)
        if out is None:
            q.put(("error", "bad_or_none_return", [[pad_value]]))
            return
        q.put(("ok", "", out))
    except Exception as e:
        q.put(("error", repr(e), [[pad_value]]))


def run_transform_with_timeout(code: str, grid, timeout_s: float, pad_value: int):
    q = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, grid, q, pad_value))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout", [[pad_value]]

    if q.empty():
        return False, "no_result", [[pad_value]]

    status, msg, out = q.get()
    out = _normalize_grid_like(out)
    if status != "ok" or out is None:
        return False, msg if msg else "error", [[pad_value]]

    return True, msg, out


# =========================
# English evolution
# =========================
def fitness_for_instructions_microbatched(
    llm,
    tokenizer,
    instructions: List[str],
    train_examples,
    cfg: RunConfig,
    mini_batch_exec: int,
) -> Tuple[List[Tuple[int, float]], List[List[Dict[str, Any]]]]:
    """
    Scores multiple English instructions efficiently by batching executor calls.

    Returns:
      fitnesses: list[(exact, cell_sum)] aligned with instructions
      traces_all: list[list[trace_per_example]] aligned with instructions
    """
    n = len(instructions)
    if n == 0:
        return [], []

    exacts = [0] * n
    cell_sums = [0.0] * n
    traces_all: List[List[Dict[str, Any]]] = [[] for _ in range(n)]

    for ex in train_examples:
        msgs_list = [build_executor_messages(instr, ex["input"]) for instr in instructions]

        payloads_all: List[str] = []
        for chunk in chunk_list(msgs_list, mini_batch_exec):
            _, ans_payloads = generate_two_phase_tagged_batch_from_messages(
                llm=llm,
                tokenizer=tokenizer,
                messages_list=chunk,
                start_tag="<answer>",
                end_tag="</answer>",
                think_cfg=cfg.executor,
                answer_cfg=cfg.executor,
            )
            payloads_all.extend(ans_payloads)

        # Align payloads with instructions
        for i in range(n):
            raw = payloads_all[i] if i < len(payloads_all) else ""
            pred_grid = extract_grid_anywhere(raw, cfg.empty_grid)
            is_match, acc = grid_accuracy(pred_grid, ex["output"], pad_value=cfg.pad_value)

            exacts[i] += int(is_match)
            cell_sums[i] += acc
            traces_all[i].append(
                {
                    "pred_grid": pred_grid,
                    "match": is_match,
                    "cell_accuracy": acc,
                    "raw_answer": raw,
                }
            )

    fitnesses = [(exacts[i], cell_sums[i]) for i in range(n)]
    return fitnesses, traces_all


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
        all_payloads.extend([p.strip() for p in instr_payloads])

    return all_payloads


def _stagewise_revise_english_candidates(
    llm,
    tokenizer,
    candidates: List[Dict[str, Any]],
    train_examples,
    cfg: RunConfig,
    stage_keeps: List[int],
    stage_rounds: List[int],
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates
    if not stage_keeps or not stage_rounds:
        return candidates
    if len(stage_keeps) != len(stage_rounds):
        raise ValueError("rev_stage_keeps and rev_stage_rounds must have same length")

    candidates.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)

    for keep_n, rounds in zip(stage_keeps, stage_rounds):
        keep_n = max(1, min(keep_n, len(candidates)))
        candidates = candidates[:keep_n]

        for _ in range(rounds):
            # 1) Build revision prompts for survivors
            rev_msgs_list = [
                build_revision_messages(c["instruction"], train_examples, c["traces"])
                for c in candidates
            ]

            # 2) Generate revised instructions micro-batched
            revised_payloads_all: List[str] = []
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
                revised_payloads_all.extend([p.strip() for p in revised_payloads])

            # 3) Score revised instructions in one batched fitness pass
            revised_fits, revised_traces_all = fitness_for_instructions_microbatched(
                llm=llm,
                tokenizer=tokenizer,
                instructions=revised_payloads_all,
                train_examples=train_examples,
                cfg=cfg,
                mini_batch_exec=cfg.batch.mini_batch_exec,
            )

            # 4) Keep best of (old vs revised) per slot
            new_candidates = []
            for old_c, new_instr, new_fit, new_traces in zip(
                candidates, revised_payloads_all, revised_fits, revised_traces_all
            ):
                new_c = {"instruction": new_instr, "fitness": new_fit, "traces": new_traces}
                if new_c["fitness"] > old_c["fitness"]:
                    new_candidates.append(new_c)
                else:
                    new_candidates.append(old_c)

            candidates = new_candidates
            candidates.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)

    return candidates


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

    # Batched scoring for initial population
    fits, traces_all = fitness_for_instructions_microbatched(
        llm=llm,
        tokenizer=tokenizer,
        instructions=instructions,
        train_examples=train_examples,
        cfg=cfg,
        mini_batch_exec=cfg.batch.mini_batch_exec,
    )

    scored = []
    for instr, fit, traces in zip(instructions, fits, traces_all):
        scored.append({"instruction": instr, "fitness": fit, "traces": traces})

    scored.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    best = scored[0]
    history = scored[:]

    if best["fitness"][0] == len(train_examples):
        return best, history

    parents = scored[: s.top_k]

    revised = []
    if s.rev_stage_keeps and s.rev_stage_rounds:
        revised = _stagewise_revise_english_candidates(
            llm=llm,
            tokenizer=tokenizer,
            candidates=[
                {"instruction": p["instruction"], "fitness": p["fitness"], "traces": p["traces"]}
                for p in parents
            ],
            train_examples=train_examples,
            cfg=cfg,
            stage_keeps=s.rev_stage_keeps,
            stage_rounds=s.rev_stage_rounds,
        )
        revised.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
        history.extend(revised)
        if revised and revised[0]["fitness"] > best["fitness"]:
            best = revised[0]

        if best["fitness"][0] == len(train_examples):
            return best, history

    # Pooling (still useful), but keep it cheaper by seeding from revised if available
    pooled = []
    if revised:
        pool_seed = [c["instruction"] for c in revised[: min(s.top_k, len(revised))]]
    else:
        pool_seed = [p["instruction"] for p in scored[: s.top_k]]

    for _ in range(s.n_pool_revisions):
        msgs = build_pool_messages(pool_seed, train_examples)
        _, payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=[msgs],
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_cfg=cfg.pool,
            answer_cfg=cfg.pool,
        )
        instr = payloads[0].strip() if payloads else ""
        # Score pooled instruction (single item, but we can still reuse batched scorer)
        p_fits, p_traces = fitness_for_instructions_microbatched(
            llm=llm,
            tokenizer=tokenizer,
            instructions=[instr],
            train_examples=train_examples,
            cfg=cfg,
            mini_batch_exec=cfg.batch.mini_batch_exec,
        )
        pooled.append({"instruction": instr, "fitness": p_fits[0], "traces": p_traces[0]})

    pooled.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    history.extend(pooled)
    if pooled and pooled[0]["fitness"] > best["fitness"]:
        best = pooled[0]

    return best, history


# =========================
# Python generation + selection
# =========================
def python_candidates_from_instruction_microbatched(
    llm,
    tokenizer,
    best_instruction: str,
    train_examples,
    n_candidates: int,
    cfg: RunConfig,
) -> List[str]:
    base_msgs = build_python_messages(best_instruction, train_examples)
    msgs_list = [base_msgs for _ in range(n_candidates)]

    all_payloads = []
    for chunk in chunk_list(msgs_list, cfg.batch.mini_batch_py):
        _, py_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=chunk,
            start_tag="<python>",
            end_tag="</python>",
            think_cfg=cfg.python,
            answer_cfg=cfg.python,
        )
        all_payloads.extend([p.strip() for p in py_payloads if p.strip()])

    return all_payloads


def score_python_on_train(code: str, train_examples, cfg: RunConfig):
    exact = 0
    cell_sum = 0.0
    errors = []
    traces = []

    for i, ex in enumerate(train_examples):
        ok, msg, pred = run_transform_with_timeout(
            code=code,
            grid=ex["input"],
            timeout_s=cfg.timeout.python_train_timeout_s,
            pad_value=cfg.pad_value,
        )
        if not ok:
            errors.append(f"Example {i}: error={msg}")
            pred = cfg.empty_grid

        is_match, acc = grid_accuracy(pred, ex["output"], pad_value=cfg.pad_value)
        exact += int(is_match)
        cell_sum += acc

        traces.append({"pred_grid": pred, "ok": ok, "msg": msg, "match": is_match, "cell_accuracy": acc})

        if ok and not is_match:
            h = len(pred) if isinstance(pred, list) else 0
            w = len(pred[0]) if (isinstance(pred, list) and pred and isinstance(pred[0], list)) else 0
            errors.append(f"Example {i}: mismatch (acc={acc:.2f}) pred_shape={h}x{w}")

    return (exact, cell_sum), errors, traces


def improve_python(llm, tokenizer, code: str, train_examples, errors_summary: str, cfg: RunConfig) -> str:
    msgs = build_python_revision_messages(code, train_examples, errors_summary)
    _, payloads = generate_two_phase_tagged_batch_from_messages(
        llm=llm,
        tokenizer=tokenizer,
        messages_list=[msgs],
        start_tag="<python>",
        end_tag="</python>",
        think_cfg=cfg.python_revision,
        answer_cfg=cfg.python_revision,
    )
    new_code = payloads[0].strip() if payloads else ""
    return new_code if new_code else code


def _stagewise_revise_python_candidates(
    llm,
    tokenizer,
    candidates: List[Dict[str, Any]],
    train_examples,
    cfg: RunConfig,
    stage_keeps: List[int],
    stage_rounds: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not candidates:
        return candidates, []
    if not stage_keeps or not stage_rounds:
        return candidates, []
    if len(stage_keeps) != len(stage_rounds):
        raise ValueError("py_rev_stage_keeps and py_rev_stage_rounds must have same length")

    history = []
    candidates.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)

    for keep_n, rounds in zip(stage_keeps, stage_rounds):
        keep_n = max(1, min(keep_n, len(candidates)))
        candidates = candidates[:keep_n]

        for _ in range(rounds):
            msgs_list = []
            for c in candidates:
                errs_summary = "\n".join(c["errors"][:10]) if c.get("errors") else "No explicit errors, only mismatches."
                msgs_list.append(build_python_revision_messages(c["code"], train_examples, errs_summary))

            revised_codes = []
            for chunk in chunk_list(msgs_list, cfg.batch.mini_batch_py):
                _, payloads = generate_two_phase_tagged_batch_from_messages(
                    llm=llm,
                    tokenizer=tokenizer,
                    messages_list=chunk,
                    start_tag="<python>",
                    end_tag="</python>",
                    think_cfg=cfg.python_revision,
                    answer_cfg=cfg.python_revision,
                )
                revised_codes.extend([p.strip() for p in payloads])

            new_candidates = []
            for old_c, new_code in zip(candidates, revised_codes):
                if not new_code:
                    new_candidates.append(old_c)
                    continue

                new_fit, new_errs, new_traces = score_python_on_train(new_code, train_examples, cfg)
                new_c = {"code": new_code, "fitness": new_fit, "errors": new_errs, "traces": new_traces}
                history.append(new_c)

                if new_c["fitness"] > old_c["fitness"]:
                    new_candidates.append(new_c)
                else:
                    new_candidates.append(old_c)

            candidates = new_candidates
            candidates.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)

    return candidates, history


def select_best_python(
    llm,
    tokenizer,
    best_instruction: str,
    train_examples,
    cfg: RunConfig,
):
    s = cfg.search

    # 1) Generate N python samples
    codes = python_candidates_from_instruction_microbatched(
        llm=llm,
        tokenizer=tokenizer,
        best_instruction=best_instruction,
        train_examples=train_examples,
        n_candidates=s.py_n_candidates,
        cfg=cfg,
    )

    # 2) Score all N
    scored = []
    for code in codes:
        fit, errs, traces = score_python_on_train(code, train_examples, cfg)
        scored.append({"code": code, "fitness": fit, "errors": errs, "traces": traces})

    if not scored:
        return {"code": "", "fitness": (0, 0.0), "errors": ["no_code_generated"], "traces": []}, []

    scored.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    best = scored[0]
    history = scored[:]

    # If no staged python schedule configured, fall back to old single-best iterative behavior
    if not (s.py_rev_stage_keeps and s.py_rev_stage_rounds):
        for _ in range(s.py_revision_steps):
            if best["fitness"][0] == len(train_examples):
                break
            errs_summary = "\n".join(best["errors"][:10]) if best["errors"] else "No explicit errors, only mismatches."
            new_code = improve_python(llm, tokenizer, best["code"], train_examples, errs_summary, cfg)
            new_fit, new_errs, new_traces = score_python_on_train(new_code, train_examples, cfg)
            cand = {"code": new_code, "fitness": new_fit, "errors": new_errs, "traces": new_traces}
            history.append(cand)
            if cand["fitness"] > best["fitness"]:
                best = cand
        return best, history

    # 3) Funnel schedule starting from top_k
    initial_pool = scored[: min(s.top_k, len(scored))]

    # Interpret schedule:
    # - If user provided keeps/rounds starting with top_k, use as-is.
    # - Else: prepend stage (keep=top_k, rounds=1) to implement:
    #   "take top_k and improve them once" as the first stage.
    stage_keeps = list(s.py_rev_stage_keeps)
    stage_rounds = list(s.py_rev_stage_rounds)

    if len(stage_keeps) != len(stage_rounds):
        raise ValueError("py_rev_stage_keeps and py_rev_stage_rounds must have same length")

    if not stage_keeps or stage_keeps[0] != s.top_k:
        stage_keeps = [s.top_k] + stage_keeps
        stage_rounds = [1] + stage_rounds

    # Run staged revision
    revised_pool, revised_history = _stagewise_revise_python_candidates(
        llm=llm,
        tokenizer=tokenizer,
        candidates=[
            {"code": c["code"], "fitness": c["fitness"], "errors": c["errors"], "traces": c["traces"]}
            for c in initial_pool
        ],
        train_examples=train_examples,
        cfg=cfg,
        stage_keeps=stage_keeps,
        stage_rounds=stage_rounds,
    )

    history.extend(revised_history)
    revised_pool.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    if revised_pool and revised_pool[0]["fitness"] > best["fitness"]:
        best = revised_pool[0]

    return best, history


# =========================
# Full pipeline
# =========================
def run_full_pipeline(
    df: pd.DataFrame,
    llm,
    tokenizer,
    logger: TeeLogger,
    cfg: RunConfig,
    max_tasks: Optional[int] = None,
):
    tasks_to_run = df if max_tasks is None else df.head(max_tasks)
    logger.print(f"Running FULL English→Python evolution pipeline on {len(tasks_to_run)} tasks...")

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

        best_py, _ = select_best_python(
            llm=llm,
            tokenizer=tokenizer,
            best_instruction=best_instruction,
            train_examples=train_examples,
            cfg=cfg,
        )

        best_code = best_py["code"]
        logger.print(f"Best Python fitness (exact, cell_sum): {best_py['fitness']}")
        logger.print("Best Python code preview:\n" + (best_code[:800] if best_code else "<EMPTY>"))

        mode = "python"
        final_pred = cfg.empty_grid

        if best_code and best_code.strip():
            ok, msg, py_pred = run_transform_with_timeout(
                code=best_code,
                grid=test_input_grid,
                timeout_s=cfg.timeout.python_test_timeout_s,
                pad_value=cfg.pad_value,
            )
            if ok:
                final_pred = py_pred
            else:
                logger.print(f"Python failed on test ({msg}); fallback to English executor.")
                mode = "english_fallback"
        else:
            mode = "english_fallback"

        if mode == "english_fallback":
            msgs = build_executor_messages(best_instruction, test_input_grid)
            _, payloads = generate_two_phase_tagged_batch_from_messages(
                llm=llm,
                tokenizer=tokenizer,
                messages_list=[msgs],
                start_tag="<answer>",
                end_tag="</answer>",
                think_cfg=cfg.executor,
                answer_cfg=cfg.executor,
            )
            final_pred = extract_grid_anywhere(payloads[0] if payloads else "", cfg.empty_grid)

        is_match, acc = grid_accuracy(final_pred, true_test_grid, pad_value=cfg.pad_value)
        logger.print(f"TEST: mode={mode} match={is_match} acc={acc:.2f}")

        out_rows.append(
            {
                "file_name": file_name,
                "final_solution": final_pred,
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

    cfg = RunConfig(
        instruction=GenPhaseConfig(
            think_tokens=6000,
            think_temp=1.1,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=1400,
            answer_temp=0.5,
            answer_top_p=1.0,
        ),
        executor=GenPhaseConfig( # low thinking!!
            think_tokens=1500,
            think_temp=0.5,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=1400,
            answer_temp=0.3,
            answer_top_p=1.0,
        ),
        python=GenPhaseConfig(
            think_tokens=4096,
            think_temp=1,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=2600,
            answer_temp=0.3,
            answer_top_p=1.0,
        ),
        python_revision=GenPhaseConfig(
            think_tokens=2048,
            think_temp=1,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=2600,
            answer_temp=0.3,
            answer_top_p=1.0,
        ),
        pool=GenPhaseConfig(
            think_tokens=2048,
            think_temp=1,
            think_top_p=0.95,
            think_top_k=20,
            answer_tokens=1400,
            answer_temp=0.3,
            answer_top_p=1.0,
        ),
        search=SearchConfig(
            n_init=40,
            top_k=20,
            n_individual_revisions=15,
            n_pool_revisions=15,
            py_n_candidates=20,
            py_revision_steps=15,

            rev_stage_keeps=[8, 5, 2],
            rev_stage_rounds=[1, 2, 4],

            py_rev_stage_keeps=[8, 5, 2],
            py_rev_stage_rounds=[1, 2, 4],
        ),
        batch=BatchConfig(
            mini_batch_init=20,
            mini_batch_revisions=15,
            mini_batch_py=20,
            mini_batch_exec=20,   # NEW
        ),
        timeout=TimeoutConfig(
            python_train_timeout_s=2.0,
            python_test_timeout_s=2.0,
        ),
        pad_value=-1,

    )

    df = create_data_frame(test_run=True, pad_value=cfg.pad_value)
    df = df.iloc[[0, 188, 5, 91, 136, 376, 10, 11, 47, 78, 169, 52, 163, 79, 238, 336, 93, 399, 210, 263, 292, 138, 316, 118, 257, 388, 394, 357, 354, 275, 233, 276, 201, 385, 383, 43, 189, 3, 303, 309]]

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"runlog_{timestamp}.txt"
    csv_path = output_dir / f"final_{timestamp}.csv"

    logger = TeeLogger(log_path)
    logger.print(f"Logging to: {log_path}")
    logger.print("RUN CONFIG:")
    logger.print(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    llm, tokenizer = init_llm_and_tokenizer(model_subdir=MODEL_SUBDIR)

    try:
        out_df = run_full_pipeline(
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
    mp.set_start_method("spawn", force=True)
    main()