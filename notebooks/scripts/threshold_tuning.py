import os
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def tune_thresholds(model_path="models/credit_model.pkl", data_path="data/raw/credit_data.csv"):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
        
    xgb = joblib.load(model_path)
    df = pd.read_csv(data_path)
    target_col = "TARGET" if "TARGET" in df.columns else "default"
    drop_cols = [c for c in ["SK_ID_CURR", target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    probs = xgb.predict_proba(X)[:, 1]
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p = precision_score(y, preds, zero_division=0)
        r = recall_score(y, preds)
        f1 = f1_score(y, preds)
        results.append({"Threshold": t, "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)})
        
    res_df = pd.DataFrame(results)
    print("=== Threshold Tuning Results ===")
    print(res_df.to_string(index=False))
    return res_df

if __name__ == "__main__":
    tune_thresholds()
