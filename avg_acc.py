import pandas as pd

# change filename if needed
df = pd.read_csv("eng_py/final_20260207_202807.csv")

# ensure numeric
df["accuracy"] = pd.to_numeric(df["accuracy"], errors="raise")

mean_acc = df["accuracy"].mean()

print(f"Average accuracy: {mean_acc:.4f}")
