import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import csv

def check_true_results(file=""):
    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    csv_file = str(output_dir) + "/evaluation_results_20251121_214514.csv"
    if not csv_file:
        print("No result CSV found.")
        return

    print(f"Loading {csv_file}")
    df = pd.read_csv(csv_file)

    task_groups = df.groupby("file_name")
    succesfull_tasks = []
    
    for task, task_df in task_groups:
        matches = task_df["match"].to_list()
        if any(match == True for match in matches):
            succesfull_tasks.append(task)

    print(succesfull_tasks)

    with open("true_tasks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(succesfull_tasks)

if __name__ == "__main__":
    check_true_results("")