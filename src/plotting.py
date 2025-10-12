import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output_qwen/p099.csv")

def avg_plot(df):
    df['timestep'] = df.groupby('file_name').cumcount() 
    avg_df = df.groupby('timestep', as_index=False)['best_accuracy'].mean() # Plot 
    plt.figure(figsize=(12,5)) 

    plt.plot(avg_df['timestep'], avg_df['best_accuracy'], linewidth=2, color="tab:blue") 
    plt.xlabel("Timestep", fontsize=12) 
    plt.ylabel("Average Best Accuracy", fontsize=12) 
    plt.title("Average Best Accuracy over Timesteps", fontsize=14) 
    plt.grid(True, linestyle="--", alpha=0.6) 
    plt.savefig("qwen_plots/360k_temp0.9_average.png", dpi=300, bbox_inches="tight")

    plt.savefig("qwen_plots/360k09p099_avg.png", dpi=300, bbox_inches="tight")

def nm_plot(df):
    df['timestep'] = df.groupby('file_name').cumcount()

    # Plot each file separately on the same figure
    plt.figure(figsize=(12,5))

    for file_name, group in df.groupby('file_name'):
        plt.plot(group['timestep'], 
                group['best_accuracy'], 
                linewidth=2, 
                label=file_name)  # no marker

    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Best Accuracy", fontsize=12)
    plt.title("Best Accuracy over Timesteps for Each File", fontsize=14)
    plt.legend(title="File Name", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("qwen_plots/360k_temp0.9p099_individual.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
    df = pd.read_csv("output_qwen/p099.csv")
    nm_plot(df)

if __name__ == "__main__":
    main()
