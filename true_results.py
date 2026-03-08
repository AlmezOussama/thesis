import pandas as pd

df = pd.read_csv("eng_py/15rev6ktokens.csv")

# extract only the needed columns
result = df[["file_name", "match"]]

# save to new csv
result.to_csv("filename_match.csv", index=False)

print(result)
