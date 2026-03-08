import pandas as pd
import matplotlib.pyplot as plt

file_1 = "/home/omeziani/thesis/dyn_ber/second_solution/final_20260228_052449.csv"
file_2 = "/home/omeziani/thesis/src/output/final_20260301_100446.csv"

# Read both
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

# Keep only needed columns
df1 = df1[["file_name", "accuracy"]]
df2 = df2[["file_name", "accuracy"]]

# Stack them (since they contain different tasks)
df = pd.concat([df1, df2], ignore_index=True)

# Make sure accuracy is numeric
df["accuracy"] = pd.to_numeric(df["accuracy"], errors="raise")

# Sort by accuracy (optional, but cleaner)
df = df.sort_values("accuracy", ascending=False)

# Plot
plt.figure()
plt.bar(df["file_name"], df["accuracy"])
plt.xticks(rotation=90)
plt.xlabel("Task")
plt.ylabel("Accuracy")
plt.title("Accuracy per ARC Task (All 40 Tasks)")
plt.tight_layout()

plt.savefig("accuracy_all_tasks.png", dpi=300)
plt.close()

print("Saved: accuracy_all_tasks.png")