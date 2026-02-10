import pandas as pd
from pathlib import Path
import csv


def check_true_results(file=""):
    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "qwen8"

    final_rows = []

    for file in output_dir.iterdir():
        if file.is_file() and file.suffix == ".csv":
            df = pd.read_csv(file)
            task_groups = df.groupby("file_name")

            for task, task_df in task_groups:
                true_rows = task_df[task_df["match"] == True]
                if len(true_rows) > 0:
                    first_true = true_rows.iloc[0]
                    sample_id = first_true["sample_id"]
                    final_rows.append([file.name, task, sample_id])

    with open("true_tasks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["csv_file", "task", "sample_id"])
        for row in final_rows:
            writer.writerow(row)


if __name__ == "__main__":
    check_true_results("")
