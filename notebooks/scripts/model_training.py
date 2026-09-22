import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc
from xgboost import XGBClassifier

def train_models(data_path="data/raw/credit_data.csv", model_output_path="models/credit_model.pkl"):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    df = pd.read_csv(data_path)
    
    target_col = "TARGET" if "TARGET" in df.columns else "default"
    drop_cols = [c for c in ["SK_ID_CURR", target_col] if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    probs = xgb.predict_proba(X_test)[:, 1]
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    
    joblib.dump(xgb, model_output_path)
    print(f"Saved model to {model_output_path}")

if __name__ == "__main__":
    train_models()
