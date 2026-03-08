from pathlib import Path
import pandas as pd


def get_table(csv):
    df = pd.read_csv(csv)

    #table = df.drop(columns=["final_solution"])
    df.to_csv("pacore/pacore_1.2.csv.csv", index=False)

    return df

def build_task_table(csv_path: str | Path) -> pd.DataFrame:
    """
    Builds one row per ARC task from a CSV like your example.

    Required columns:
      - file_name
      - sample_id
      - match (bool)
      - cell_accuracy (float, 0–100)
    """
    df = pd.read_csv(csv_path)

    required = {"file_name", "sample_id", "match", "cell_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["match"] = df["match"].astype(bool)
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="raise")
    df["cell_accuracy"] = pd.to_numeric(df["cell_accuracy"], errors="raise")

    def first_success_k(g):
        s = g.loc[g["match"], "sample_id"]
        return int(s.min()) if len(s) else pd.NA

    table = (
        df.groupby("file_name", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_samples": int(len(g)),
                    "total_successes": int(g["match"].sum()),
                    "success_rate": float(g["match"].mean()),
                    "first_success_k": first_success_k(g),
                    "best_cell_accuracy": float(g["cell_accuracy"].max()),
                    "mean_cell_accuracy": float(g["cell_accuracy"].mean()),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values(
            ["total_successes", "best_cell_accuracy"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return table


if __name__ == "__main__":
    # table = build_task_table("qwen8/qwen8b_twophase_20260208_121026_tT1.1_tP0.95_tK20_aT0.2_aP1.0_k102.csv")
    # table.to_csv("task_sum_Two_phase.csv", index=False)
    table = get_table("pacore/ev_1.2_0_120.csv") 
    print(table.head(10).to_string(index=False))
