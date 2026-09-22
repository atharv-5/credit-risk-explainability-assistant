import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def run_shap_analysis(model_path="models/credit_model.pkl", data_path="data/raw/credit_data.csv"):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
        
    xgb = joblib.load(model_path)
    df = pd.read_csv(data_path)
    target_col = "TARGET" if "TARGET" in df.columns else "default"
    drop_cols = [c for c in ["SK_ID_CURR", target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    sample_size = min(2000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"Computed SHAP values shape: {shap_values.shape}")
    return explainer, shap_values

if __name__ == "__main__":
    run_shap_analysis()
