"""
ARC English→Python Evolution Pipeline (two-phase thinking, micro-batched generation)

Micro-batching added for:
- n_init instruction candidates (per task) via mini_batch_size_init
- independent revisions (per task) via mini_batch_size_revisions
- py_n_candidates python candidates (per task) via mini_batch_size_py

Notes:
- Batching here batches only LLM generation. Python execution remains CPU and timeout-based.
- Grid padding uses -1; missing/invalid grid returns [[-1]]
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

import os
import re
import ast
import json
import builtins
import multiprocessing as mp

import numpy as np
import pandas as pd


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


def create_data_frame(test_run=True):
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
            true_out = sol_list[test_idx] if test_idx < len(sol_list) else [[-1]]
            test_output = [{"output": true_out}]
        else:
            test_output = [{"output": [[-1]]}]

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
    correct_percentage = (correct_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    is_correct = (correct_pixels == total_pixels)
    return is_correct, correct_percentage


def extract_grid_anywhere(text: str):
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
        return [[-1]]

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

    return [[-1]]


def messages_to_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _ensure_wrapped_think(text: str) -> str:
    t = text.strip()
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
    think_max_tokens: int,
    think_temp: float,
    think_top_p: float = 0.95,
    think_top_k: int = 20,
    answer_max_tokens: int = 2048,
    answer_temp: float = 0.2,
    answer_top_p: float = 1.0,
) -> Tuple[List[str], List[str]]:
    from vllm import SamplingParams

    base_prompts = [messages_to_prompt(tokenizer, msgs) for msgs in messages_list]

    think_params = SamplingParams(
        temperature=think_temp,
        top_p=think_top_p,
        top_k=think_top_k,
        max_tokens=think_max_tokens,
        stop=[start_tag],
    )
    think_outs = llm.generate(base_prompts, think_params)

    thinks = []
    for o in think_outs:
        txt = o.outputs[0].text.strip() if o.outputs else ""
        thinks.append(_ensure_wrapped_think(txt))

    phase2_prompts = [bp + th + start_tag for bp, th in zip(base_prompts, thinks)]

    answer_params = SamplingParams(
        temperature=answer_temp,
        top_p=answer_top_p,
        max_tokens=answer_max_tokens,
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
        "Translate the instruction literally into code.\n"
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


def _normalize_grid_like(out):
    if out is None:
        return None
    if not isinstance(out, list):
        return None
    if not all(isinstance(r, list) for r in out):
        return None
    return out


def _worker_exec(code: str, grid, q: mp.Queue):
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
            q.put(("error", "no_transform_defined", [[-1]]))
            return

        out = env["transform"](grid)
        out = _normalize_grid_like(out)
        if out is None:
            q.put(("error", "bad_or_none_return", [[-1]]))
            return

        q.put(("ok", "", out))
    except Exception as e:
        q.put(("error", repr(e), [[-1]]))


def run_transform_with_timeout(code: str, grid, timeout_s=0.8):
    q = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, grid, q))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout", [[-1]]

    if q.empty():
        return False, "no_result", [[-1]]

    status, msg, out = q.get()
    out = _normalize_grid_like(out)
    if status != "ok" or out is None:
        return False, msg if msg else "error", [[-1]]

    return True, msg, out


def fitness_for_instruction(
    llm,
    tokenizer,
    instruction: str,
    train_examples,
    executor_think_tokens: int,
    think_temp: float = 0.8,
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
            think_max_tokens=executor_think_tokens,
            think_temp=think_temp,
            answer_max_tokens=1200,
            answer_temp=0.2,
        )
        pred_grid = extract_grid_anywhere(ans_payloads[0])
        is_match, acc = grid_accuracy(pred_grid, ex["output"], pad_value=-1)
        exact += int(is_match)
        cell_sum += acc
        traces.append(
            {
                "pred_grid": pred_grid,
                "match": is_match,
                "cell_accuracy": acc,
                "raw_answer": ans_payloads[0],
            }
        )

    return (exact, cell_sum), traces


def generate_instruction_candidates_microbatched(
    llm,
    tokenizer,
    train_examples,
    n_candidates: int,
    instruction_think_tokens: int,
    think_temp: float,
    mini_batch_size: int,
) -> List[str]:
    base_msgs = build_instruction_messages(train_examples)
    msgs_list = [base_msgs for _ in range(n_candidates)]

    all_payloads = []
    for chunk in chunk_list(msgs_list, mini_batch_size):
        _, instr_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=chunk,
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_max_tokens=instruction_think_tokens,
            think_temp=think_temp,
            answer_max_tokens=1400,
            answer_temp=0.2,
        )
        all_payloads.extend([p.strip() for p in instr_payloads])

    return all_payloads


def evolve_english_instruction_for_task(
    llm,
    tokenizer,
    train_examples,
    n_init: int = 12,
    top_k: int = 5,
    n_individual_revisions: int = 5,
    n_pool_revisions: int = 3,
    instruction_think_tokens: int = 4096,
    executor_think_tokens: int = 2048,
    think_temp_init: float = 1.0,
    think_temp_exec: float = 0.8,
    mini_batch_size_init: int = 10,
    mini_batch_size_revisions: int = 10,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    instructions = generate_instruction_candidates_microbatched(
        llm=llm,
        tokenizer=tokenizer,
        train_examples=train_examples,
        n_candidates=n_init,
        instruction_think_tokens=instruction_think_tokens,
        think_temp=think_temp_init,
        mini_batch_size=mini_batch_size_init,
    )

    scored = []
    for instr in instructions:
        fit, traces = fitness_for_instruction(
            llm=llm,
            tokenizer=tokenizer,
            instruction=instr,
            train_examples=train_examples,
            executor_think_tokens=executor_think_tokens,
            think_temp=think_temp_exec,
        )
        scored.append({"instruction": instr, "fitness": fit, "traces": traces})

    scored.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    best = scored[0]

    if best["fitness"][0] == len(train_examples):
        return best, scored

    parents = scored[:top_k]
    n_rev = min(n_individual_revisions, len(parents))

    rev_msgs_list = []
    for p in parents[:n_rev]:
        rev_msgs_list.append(build_revision_messages(p["instruction"], train_examples, p["traces"]))

    revised_payloads_all = []
    for chunk in chunk_list(rev_msgs_list, mini_batch_size_revisions):
        _, revised_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=chunk,
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_max_tokens=instruction_think_tokens,
            think_temp=0.9,
            answer_max_tokens=1400,
            answer_temp=0.2,
        )
        revised_payloads_all.extend([p.strip() for p in revised_payloads])

    revised = []
    for instr in revised_payloads_all:
        instr = instr.strip()
        fit, traces = fitness_for_instruction(
            llm=llm,
            tokenizer=tokenizer,
            instruction=instr,
            train_examples=train_examples,
            executor_think_tokens=executor_think_tokens,
            think_temp=think_temp_exec,
        )
        revised.append({"instruction": instr, "fitness": fit, "traces": traces})

    revised.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    if revised and revised[0]["fitness"] > best["fitness"]:
        best = revised[0]

    if best["fitness"][0] == len(train_examples):
        return best, scored + revised

    pooled = []
    top_instrs = [p["instruction"] for p in scored[:top_k]]
    for _ in range(n_pool_revisions):
        msgs = build_pool_messages(top_instrs, train_examples)
        _, payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=[msgs],
            start_tag="<instruction>",
            end_tag="</instruction>",
            think_max_tokens=instruction_think_tokens,
            think_temp=0.8,
            answer_max_tokens=1400,
            answer_temp=0.2,
        )
        instr = payloads[0].strip()
        fit, traces = fitness_for_instruction(
            llm=llm,
            tokenizer=tokenizer,
            instruction=instr,
            train_examples=train_examples,
            executor_think_tokens=executor_think_tokens,
            think_temp=think_temp_exec,
        )
        pooled.append({"instruction": instr, "fitness": fit, "traces": traces})

    pooled.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    if pooled and pooled[0]["fitness"] > best["fitness"]:
        best = pooled[0]

    return best, scored + revised + pooled


def python_candidates_from_instruction_microbatched(
    llm,
    tokenizer,
    best_instruction: str,
    train_examples,
    n_candidates: int,
    python_think_tokens: int,
    think_temp: float,
    mini_batch_size: int,
) -> List[str]:
    base_msgs = build_python_messages(best_instruction, train_examples)
    msgs_list = [base_msgs for _ in range(n_candidates)]

    all_payloads = []
    for chunk in chunk_list(msgs_list, mini_batch_size):
        _, py_payloads = generate_two_phase_tagged_batch_from_messages(
            llm=llm,
            tokenizer=tokenizer,
            messages_list=chunk,
            start_tag="<python>",
            end_tag="</python>",
            think_max_tokens=python_think_tokens,
            think_temp=think_temp,
            answer_max_tokens=2600,
            answer_temp=0.15,
        )
        all_payloads.extend([p.strip() for p in py_payloads if p.strip()])

    return all_payloads


def score_python_on_train(code: str, train_examples, timeout_s=5.0):
    exact = 0
    cell_sum = 0.0
    errors = []
    traces = []

    for i, ex in enumerate(train_examples):
        ok, msg, pred = run_transform_with_timeout(code, ex["input"], timeout_s=timeout_s)
        if not ok:
            errors.append(f"Example {i}: error={msg}")
            pred = [[-1]]

        is_match, acc = grid_accuracy(pred, ex["output"], pad_value=-1)
        exact += int(is_match)
        cell_sum += acc

        traces.append(
            {
                "pred_grid": pred,
                "ok": ok,
                "msg": msg,
                "match": is_match,
                "cell_accuracy": acc,
            }
        )

        if ok and not is_match:
            h = len(pred) if isinstance(pred, list) else 0
            w = len(pred[0]) if (isinstance(pred, list) and pred and isinstance(pred[0], list)) else 0
            errors.append(f"Example {i}: mismatch (acc={acc:.2f}) pred_shape={h}x{w}")

    return (exact, cell_sum), errors, traces


def improve_python(
    llm,
    tokenizer,
    code: str,
    train_examples,
    errors_summary: str,
    python_think_tokens: int,
) -> str:
    msgs = build_python_revision_messages(code, train_examples, errors_summary)
    _, payloads = generate_two_phase_tagged_batch_from_messages(
        llm=llm,
        tokenizer=tokenizer,
        messages_list=[msgs],
        start_tag="<python>",
        end_tag="</python>",
        think_max_tokens=python_think_tokens,
        think_temp=0.75,
        answer_max_tokens=2600,
        answer_temp=0.15,
    )
    new_code = payloads[0].strip() if payloads else ""
    return new_code if new_code else code


def select_best_python(
    llm,
    tokenizer,
    best_instruction: str,
    train_examples,
    n_candidates: int,
    n_revision_steps: int,
    python_think_tokens: int,
    timeout_s: float,
    mini_batch_size_py: int,
):
    candidates = python_candidates_from_instruction_microbatched(
        llm=llm,
        tokenizer=tokenizer,
        best_instruction=best_instruction,
        train_examples=train_examples,
        n_candidates=n_candidates,
        python_think_tokens=python_think_tokens,
        think_temp=0.85,
        mini_batch_size=mini_batch_size_py,
    )

    scored = []
    for code in candidates:
        fit, errs, traces = score_python_on_train(code, train_examples, timeout_s=timeout_s)
        scored.append({"code": code, "fitness": fit, "errors": errs, "traces": traces})

    if not scored:
        return {"code": "", "fitness": (0, 0.0), "errors": ["no_code_generated"], "traces": []}, []

    scored.sort(key=lambda x: (x["fitness"][0], x["fitness"][1]), reverse=True)
    best = scored[0]
    history = scored[:]

    for _ in range(n_revision_steps):
        if best["fitness"][0] == len(train_examples):
            break
        errs_summary = "\n".join(best["errors"][:10]) if best["errors"] else "No explicit errors, only mismatches."
        new_code = improve_python(
            llm=llm,
            tokenizer=tokenizer,
            code=best["code"],
            train_examples=train_examples,
            errors_summary=errs_summary,
            python_think_tokens=python_think_tokens,
        )
        new_fit, new_errs, new_traces = score_python_on_train(new_code, train_examples, timeout_s=timeout_s)
        cand = {"code": new_code, "fitness": new_fit, "errors": new_errs, "traces": new_traces}
        history.append(cand)
        if cand["fitness"] > best["fitness"]:
            best = cand

    return best, history


def run_full_pipeline(
    df: pd.DataFrame,
    llm,
    tokenizer,
    logger: TeeLogger,
    max_tasks: Optional[int] = None,
    n_init: int = 12,
    top_k: int = 5,
    n_individual_revisions: int = 5,
    n_pool_revisions: int = 3,
    instruction_think_tokens: int = 4096,
    executor_think_tokens: int = 2048,
    python_think_tokens: int = 2048,
    py_n_candidates: int = 10,
    py_revision_steps: int = 5,
    python_train_timeout_s: float = 5.0,
    python_test_timeout_s: float = 10.0,
    mini_batch_size_init: int = 10,
    mini_batch_size_revisions: int = 10,
    mini_batch_size_py: int = 10,
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
            n_init=n_init,
            top_k=top_k,
            n_individual_revisions=n_individual_revisions,
            n_pool_revisions=n_pool_revisions,
            instruction_think_tokens=instruction_think_tokens,
            executor_think_tokens=executor_think_tokens,
            think_temp_init=1.2,
            think_temp_exec=0.8,
            mini_batch_size_init=mini_batch_size_init,
            mini_batch_size_revisions=mini_batch_size_revisions,
        )

        best_instruction = best_eng["instruction"]
        best_eng_fit = best_eng["fitness"]
        logger.print(f"Best English fitness (exact, cell_sum): {best_eng_fit}")
        logger.print("Best instruction:\n" + best_instruction[:1200] + ("..." if len(best_instruction) > 1200 else ""))

        best_py, _ = select_best_python(
            llm=llm,
            tokenizer=tokenizer,
            best_instruction=best_instruction,
            train_examples=train_examples,
            n_candidates=py_n_candidates,
            n_revision_steps=py_revision_steps,
            python_think_tokens=python_think_tokens,
            timeout_s=python_train_timeout_s,
            mini_batch_size_py=mini_batch_size_py,
        )

        best_code = best_py["code"]
        logger.print(f"Best Python fitness (exact, cell_sum): {best_py['fitness']}")
        logger.print("Best Python code preview:\n" + (best_code[:800] if best_code else "<EMPTY>"))

        final_pred = [[-1]]
        mode = "python"

        if best_code and best_code.strip():
            ok, msg, py_pred = run_transform_with_timeout(best_code, test_input_grid, timeout_s=python_test_timeout_s)
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
                think_max_tokens=executor_think_tokens,
                think_temp=0.7,
                answer_max_tokens=1400,
                answer_temp=0.2,
            )
            final_pred = extract_grid_anywhere(payloads[0] if payloads else "")

        is_match, acc = grid_accuracy(final_pred, true_test_grid, pad_value=-1)
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


def main():
    import torch

    MODEL_SUBDIR = "q3_think_f8"
    CUDA_VISIBLE = "0"

    INSTRUCTION_THINK_TOKENS = 4096
    EXECUTOR_THINK_TOKENS = 2048
    PYTHON_THINK_TOKENS = 2048

    N_INIT = 20
    TOP_K = 10
    N_INDIV_REVS = 5
    N_POOL_REVS = 5

    PY_N_CANDS = 20
    PY_REV_STEPS = 5

    PY_TRAIN_TIMEOUT_S = 5.0
    PY_TEST_TIMEOUT_S = 10.0

    MINI_BATCH_INIT = 20
    MINI_BATCH_REVS = 5
    MINI_BATCH_PY = 20

    df = create_data_frame(test_run=True)
    df = df.iloc[[0]]

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"runlog_{timestamp}.txt"
    csv_path = output_dir / f"final_{timestamp}.csv"

    logger = TeeLogger(log_path)
    logger.print(f"Logging to: {log_path}")

    llm, tokenizer = init_llm_and_tokenizer(model_subdir=MODEL_SUBDIR, cuda_visible=CUDA_VISIBLE)

    try:
        out_df = run_full_pipeline(
            df=df,
            llm=llm,
            tokenizer=tokenizer,
            logger=logger,
            max_tasks=len(df),
            n_init=N_INIT,
            top_k=TOP_K,
            n_individual_revisions=N_INDIV_REVS,
            n_pool_revisions=N_POOL_REVS,
            instruction_think_tokens=INSTRUCTION_THINK_TOKENS,
            executor_think_tokens=EXECUTOR_THINK_TOKENS,
            python_think_tokens=PYTHON_THINK_TOKENS,
            py_n_candidates=PY_N_CANDS,
            py_revision_steps=PY_REV_STEPS,
            python_train_timeout_s=PY_TRAIN_TIMEOUT_S,
            python_test_timeout_s=PY_TEST_TIMEOUT_S,
            mini_batch_size_init=MINI_BATCH_INIT,
            mini_batch_size_revisions=MINI_BATCH_REVS,
            mini_batch_size_py=MINI_BATCH_PY,
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
