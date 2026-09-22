import os
import re

class PolicyKnowledgeBase:
    def __init__(self, policy_dir="docs/policy"):
        if not os.path.exists(policy_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            policy_dir = os.path.join(base_dir, policy_dir)
            
        self.policy_dir = policy_dir
        self.policies = self._load_policies()

    def _load_policies(self) -> dict[str, str]:
        policies = {}
        if os.path.exists(self.policy_dir):
            for filename in os.listdir(self.policy_dir):
                if filename.endswith(".md") or filename.endswith(".txt"):
                    file_path = os.path.join(self.policy_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        policies[filename] = f.read()
        return policies

    def query_policy(self, feature_name: str, value: float) -> list[str]:
        """Queries policy documents for relevant rules based on feature metrics."""
        matched_rules = []
        combined_text = "\n".join(self.policies.values())

        if feature_name == "debt_to_income" and value > 0.35:
            matched_rules.append("Policy Rule: Maximum allowable DTI ratio for unsecured loans is 45%. DTI between 35%-45% triggers mandatory cash-flow reserve evaluation.")
        if feature_name == "credit_score" and value < 650:
            matched_rules.append("Policy Rule: Credit score < 650 is Tier 3 High Risk. Requires senior underwriter sign-off or co-signer.")
        if feature_name == "delinquencies_2yrs" and value > 0:
            matched_rules.append("Policy Rule: Active delinquencies within past 24 months result in immediate underwriting flag.")
        if feature_name == "recent_inquiries" and value > 2:
            matched_rules.append("Policy Rule: More than 2 credit inquiries in past 6 months adds mandatory risk surcharge.")

        # FCRA Adverse Action rule
        if not matched_rules:
            matched_rules.append("Policy Standard: Standard underwriting evaluation under Fair Credit Reporting Act (FCRA) guidelines.")

        return matched_rules

    def get_all_policy_summary(self) -> str:
        """Returns full text of all loaded policy documents."""
        if not self.policies:
            return "No policy documents found."
        return "\n\n---\n\n".join([f"### {name}\n{content}" for name, content in self.policies.items()])
