import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(csv_path=None):
    output_dir = Path(__file__).resolve().parent / "output"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csv_files = sorted(output_dir.glob("evaluation_results_*.csv"))
        if not csv_files:
            print("No evaluation CSV found in output folder.")
            return
        csv_path = csv_files[-1]  
    print(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    for task_name, group in df.groupby("file_name"):
        plt.plot(group["sample_id"], group["cell_accuracy"], marker="o", label=task_name)

    plt.title("Cell Accuracy per Task over Samples")
    plt.xlabel("Sample ID (generation step)")
    plt.ylabel("Cell Accuracy (%)")
    plt.legend(fontsize="small", loc="lower right", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    task_plot_path = plots_dir / "accuracy_per_task.png"
    plt.savefig(task_plot_path)
    plt.close()
    print(f"Saved plot: {task_plot_path}")

    avg_accuracy = df.groupby("sample_id")["cell_accuracy"].mean()

    plt.figure(figsize=(8, 5))
    plt.plot(avg_accuracy.index, avg_accuracy.values, marker="o", color="tab:blue")
    plt.title("Average Accuracy over All Tasks per Sample Step")
    plt.xlabel("Sample ID (generation step)")
    plt.ylabel("Average Cell Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    avg_plot_path = plots_dir / "average_accuracy_over_time.png"
    plt.savefig(avg_plot_path)
    plt.close()
    print(f"Saved plot: {avg_plot_path}")


if __name__ == "__main__":
    plot_results()
