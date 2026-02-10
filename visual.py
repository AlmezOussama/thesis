from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_task_summary_table(
    csv_path: str | Path,
    task_col_candidates=("filename", "file_name", "task", "task_id"),
    acc_col_candidates=("accuracy", "acc", "match"),
    sample_col_candidates=("sample_id", "sample_idx", "sample", "k", "n"),
    success_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Builds one row per task with summary metrics.

    Expected input: one row per (task, sample).
    Columns needed:
      - task identifier (e.g., filename)
      - accuracy or boolean match
      - sample index (optional but recommended). If missing, row order within each task is used.

    Output columns:
      - task
      - n_samples
      - best_acc
      - mean_acc
      - total_successes
      - success_rate
      - first_success_k
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    task_col = pick_col(task_col_candidates)
    acc_col = pick_col(acc_col_candidates)
    sample_col = pick_col(sample_col_candidates)

    if task_col is None:
        raise ValueError(
            f"No task column found. Tried: {task_col_candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    if acc_col is None:
        raise ValueError(
            f"No accuracy/match column found. Tried: {acc_col_candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalize accuracy column to numeric in [0, 1] when possible
    acc = df[acc_col]
    if acc.dtype == bool:
        df["_acc"] = acc.astype(float)
    else:
        # Handles strings like "True"/"False" or numbers
        if acc.dtype == object:
            acc_lower = acc.astype(str).str.lower()
            if acc_lower.isin(["true", "false"]).all():
                df["_acc"] = acc_lower.map({"true": 1.0, "false": 0.0}).astype(float)
            else:
                df["_acc"] = pd.to_numeric(acc, errors="coerce")
        else:
            df["_acc"] = pd.to_numeric(acc, errors="coerce")

    if df["_acc"].isna().any():
        bad = df[df["_acc"].isna()].head(5)
        raise ValueError(
            "Could not convert some accuracy values to numeric. "
            f"Example problematic rows:\n{bad[[task_col, acc_col]].to_string(index=False)}"
        )

    # Define sample order k
    if sample_col is not None:
        df["_k"] = pd.to_numeric(df[sample_col], errors="coerce")
        if df["_k"].isna().any():
            # Fall back to per-task row order if sample col is messy
            df = df.sort_values([task_col]).copy()
            df["_k"] = df.groupby(task_col).cumcount() + 1
        else:
            # If sample ids are 0-based, convert to 1-based for readability
            if df["_k"].min() == 0:
                df["_k"] = df["_k"] + 1
    else:
        df = df.sort_values([task_col]).copy()
        df["_k"] = df.groupby(task_col).cumcount() + 1

    df["_success"] = df["_acc"] >= float(success_threshold)

    def first_success_k(g: pd.DataFrame) -> int | pd.NA:
        s = g.loc[g["_success"], "_k"]
        return int(s.min()) if len(s) else pd.NA

    summary = (
        df.groupby(task_col, as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_samples": int(g.shape[0]),
                    "best_acc": float(g["_acc"].max()),
                    "mean_acc": float(g["_acc"].mean()),
                    "total_successes": int(g["_success"].sum()),
                    "success_rate": float(g["_success"].mean()),
                    "first_success_k": first_success_k(g),
                }
            )
        )
        .reset_index(drop=True)
        .rename(columns={task_col: "task"})
        .sort_values(["total_successes", "best_acc", "mean_acc"], ascending=False)
        .reset_index(drop=True)
    )

    return summary


def build_summary_for_directory(
    input_dir: str | Path,
    pattern: str = "*.csv",
    out_path: str | Path = "task_summary_all_runs.csv",
) -> pd.DataFrame:
    """
    For multiple CSV files (e.g., different runs/strategies), builds a combined table.
    Adds a 'run_file' column to identify which CSV each row came from.
    """
    input_dir = Path(input_dir)
    rows = []
    for p in sorted(input_dir.glob(pattern)):
        try:
            s = build_task_summary_table(p)
            s.insert(0, "run_file", p.name)
            rows.append(s)
        except Exception as e:
            raise RuntimeError(f"Failed on {p}: {e}") from e

    if not rows:
        raise ValueError(f"No files matched {pattern} in {input_dir}")

    out = pd.concat(rows, ignore_index=True)
    out_path = Path(out_path)
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    # Single CSV -> single task summary table
    # summary = build_task_summary_table("results.csv")
    # summary.to_csv("task_summary.csv", index=False)

    # Directory of CSVs -> combined table
    combined = build_summary_for_directory(
        input_dir="pacore",               # change to your folder, e.g. "outputs/pacore"
        pattern="*.csv",
        out_path="task_summary_all_runs.csv",
    )
    print(combined.head(20).to_string(index=False))
