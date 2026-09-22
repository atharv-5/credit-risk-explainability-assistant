import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

class ExplainabilityEngine:
    def __init__(self, model_path="models/credit_model.pkl", feature_names_path="models/feature_names.json"):
        # Resolve path relative to current working dir or package root
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)
            feature_names_path = os.path.join(base_dir, feature_names_path)
            
        self.model = joblib.load(model_path)
        with open(feature_names_path, "r") as f:
            self.feature_names = json.load(f)
            
        self.explainer = shap.TreeExplainer(self.model)

    def predict_risk(self, input_data: dict) -> tuple[float, int]:
        """Returns default probability (0.0 to 1.0) and binary decision (1 = High Risk, 0 = Low Risk)."""
        df = pd.DataFrame([input_data])[self.feature_names]
        prob = float(self.model.predict_proba(df)[0, 1])
        prediction = int(prob >= 0.35)  # 35% threshold for credit risk flag
        return prob, prediction

    def get_shap_explanation(self, input_data: dict) -> dict:
        """Computes SHAP values and identifies top adverse (risk-increasing) and positive (risk-reducing) features."""
        df = pd.DataFrame([input_data])[self.feature_names]
        shap_values = self.explainer(df)
        
        values = shap_values.values[0]
        base_value = float(shap_values.base_values[0])
        
        feature_impacts = []
        for name, val, input_val in zip(self.feature_names, values, df.iloc[0]):
            feature_impacts.append({
                "feature": name,
                "value": float(input_val),
                "shap_value": float(val),
                "impact": "Increases Risk" if val > 0 else "Decreases Risk"
            })
            
        # Sort by absolute SHAP impact
        feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        adverse_factors = [f for f in feature_impacts if f["shap_value"] > 0]
        favorable_factors = [f for f in feature_impacts if f["shap_value"] < 0]
        
        return {
            "base_value": base_value,
            "feature_impacts": feature_impacts,
            "adverse_factors": adverse_factors,
            "favorable_factors": favorable_factors
        }

    def generate_counterfactuals(self, input_data: dict) -> list[str]:
        """Generates actionable recommendations for applicants to lower risk score."""
        explanation = self.get_shap_explanation(input_data)
        recommendations = []
        
        for factor in explanation["adverse_factors"]:
            feat = factor["feature"]
            val = factor["value"]
            
            if feat == "debt_to_income" and val > 0.35:
                recommendations.append(f"Reduce Debt-to-Income ratio from {val*100:.1f}% to under 35% by paying off existing loans.")
            elif feat == "credit_score" and val < 680:
                recommendations.append(f"Improve credit score (currently {val:.0f}) to at least 680 by ensuring zero missed payments.")
            elif feat == "delinquencies_2yrs" and val > 0:
                recommendations.append(f"Maintain 24 consecutive months without delinquencies (currently {val:.0f}).")
            elif feat == "recent_inquiries" and val > 1:
                recommendations.append(f"Avoid applying for new credit lines to decrease recent inquiries (currently {val:.0f}).")
            elif feat == "loan_amount":
                recommendations.append("Consider requesting a lower loan amount or offering a higher down payment.")

        if not recommendations:
            recommendations.append("Profile is low risk. Maintain current financial health and timely payments.")

        return recommendations
