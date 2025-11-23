import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_task_results(output_dir, prefix=""):

    src_dir = Path(__file__).resolve().parent
    output_dir = src_dir / "output"
    csv_file = str(output_dir) + "/evaluation_results_20251114_034044.csv"
    if not csv_file:
        print("No result CSV found.")
        return

    print(f"Loading {csv_file}")
    df = pd.read_csv(csv_file)

    # Ensure numeric accuracy
    df["cell_accuracy"] = pd.to_numeric(df["cell_accuracy"], errors="coerce").fillna(0)

    # Group by task
    task_groups = df.groupby("file_name")
    all_accuracies = []

    for task, task_df in task_groups:
        task_df = task_df.sort_values("sample_id").reset_index(drop=True)
        accuracies = task_df["cell_accuracy"].to_list()
        all_accuracies.append(accuracies)

        # Plot for this task
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
        plt.title(f"Accuracy per Sample — {task}")
        plt.xlabel("Sample ID")
        plt.ylabel("Cell Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_path = Path(output_dir) / f"{task}_accuracy_curve.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path.name}")

    # Compute average accuracy per sample index across all tasks
    max_len = max(len(x) for x in all_accuracies)
    padded = [x + [np.nan] * (max_len - len(x)) for x in all_accuracies]
    avg_curve = np.nanmean(padded, axis=0)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(avg_curve) + 1), avg_curve, marker='o', color='purple')
    plt.title("Average Accuracy Across All Tasks")
    plt.xlabel("Sample Index")
    plt.ylabel("Average Cell Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    avg_path = Path(output_dir) / "average_accuracy_all_tasks.png"
    plt.savefig(avg_path)
    plt.close()
    print(f"Saved {avg_path.name}")


if __name__ == "__main__":
    plot_task_results("src/output", prefix="eval")
