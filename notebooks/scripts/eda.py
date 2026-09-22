import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def run_eda(data_path="data/raw/cleaned_application_train.csv"):
    df = pd.read_csv(data_path)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("=== Data types & non-null counts ===")
    print(df.info())
    
    print("\n=== Target distribution ===")
    counts = df["TARGET"].value_counts()
    pct = df["TARGET"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "pct": pct.round(2)}))
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["TARGET"].drop("TARGET").sort_values(key=abs, ascending=False)
    print("\n=== Top 15 features correlated with TARGET ===")
    print(corr.head(15))
    return df

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cleaned_application_train.csv"
    run_eda(path)
