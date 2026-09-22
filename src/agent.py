from src.explain import ExplainabilityEngine
from src.knowledge_base import PolicyKnowledgeBase

class CreditRiskAgent:
    def __init__(self):
        self.explain_engine = ExplainabilityEngine()
        self.knowledge_base = PolicyKnowledgeBase()

    def evaluate_applicant(self, applicant_data: dict) -> dict:
        """
        Full evaluation pipeline:
        1. Predict risk probability & binary classification.
        2. Extract SHAP feature attributions.
        3. Query policy knowledge base for relevant compliance rules.
        4. Synthesize structured report & actionable counterfactuals.
        """
        prob_default, risk_flag = self.explain_engine.predict_risk(applicant_data)
        shap_analysis = self.explain_engine.get_shap_explanation(applicant_data)
        counterfactuals = self.explain_engine.generate_counterfactuals(applicant_data)

        # Retrieve relevant policies for top adverse factors
        relevant_policies = []
        for factor in shap_analysis["adverse_factors"]:
            rules = self.knowledge_base.query_policy(factor["feature"], factor["value"])
            for rule in rules:
                if rule not in relevant_policies:
                    relevant_policies.append(rule)

        # Risk tier classification
        if prob_default < 0.20:
            risk_tier = "Tier 1 - Low Risk (Preferred Eligibility)"
            decision = "APPROVED"
            decision_color = "green"
        elif prob_default < 0.40:
            risk_tier = "Tier 2 - Moderate Risk (Underwriter Review)"
            decision = "REVIEW REQUIRED"
            decision_color = "orange"
        else:
            risk_tier = "Tier 3 - High Risk (High Default Likelihood)"
            decision = "REJECTED / ADVERSE ACTION"
            decision_color = "red"

        # Generate markdown report
        report = f"""
### 📋 Credit Decision & Compliance Report

**Decision Status**: `{decision}`  
**Default Probability**: `{prob_default * 100:.1f}%`  
**Risk Category**: `{risk_tier}`  

---

#### 🔍 Key Risk Drivers (SHAP Attribution)
"""
        if shap_analysis["adverse_factors"]:
            for factor in shap_analysis["adverse_factors"]:
                report += f"- **{factor['feature'].replace('_', ' ').title()}** (`{factor['value']}`): Increased risk by `+{factor['shap_value']:.3f}` SHAP points.\n"
        else:
            report += "- No major adverse risk factors detected.\n"

        report += "\n#### 🛡️ Favorable Factors\n"
        if shap_analysis["favorable_factors"]:
            for factor in shap_analysis["favorable_factors"]:
                report += f"- **{factor['feature'].replace('_', ' ').title()}** (`{factor['value']}`): Decreased risk by `{factor['shap_value']:.3f}` SHAP points.\n"
        else:
            report += "- Limited favorable credit factors found.\n"

        report += "\n#### 📜 Regulatory & Policy Compliance Citations\n"
        if relevant_policies:
            for policy in relevant_policies:
                report += f"- {policy}\n"
        else:
            report += "- Standard underwriting guidelines applied.\n"

        report += "\n#### 💡 Actionable Guidance & Remediation Plan\n"
        for rec in counterfactuals:
            report += f"1. {rec}\n"

        return {
            "decision": decision,
            "decision_color": decision_color,
            "prob_default": prob_default,
            "risk_tier": risk_tier,
            "shap_analysis": shap_analysis,
            "relevant_policies": relevant_policies,
            "counterfactuals": counterfactuals,
            "markdown_report": report
        }
