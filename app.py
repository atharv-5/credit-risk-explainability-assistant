import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent import CreditRiskAgent

st.set_page_config(
    page_title="Credit Risk Explainability System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return CreditRiskAgent()

agent = load_agent()

st.markdown('<div class="main-title">💳 Credit Risk Explainability & AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">XAI (SHAP) + Policy RAG + Decision Report Synthesis</div>', unsafe_allow_html=True)

# Sidebar input controls
st.sidebar.header("📋 Applicant Financial Profile")

age = st.sidebar.slider("Age", 18, 80, 35)
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
credit_score = st.sidebar.slider("Credit Score (FICO)", 300, 850, 670)
debt_to_income = st.sidebar.slider("Debt-to-Income (DTI) Ratio", 0.05, 0.80, 0.38, step=0.01)
delinquencies_2yrs = st.sidebar.selectbox("Delinquencies in Past 2 Years", [0, 1, 2, 3, 4], index=0)
recent_inquiries = st.sidebar.selectbox("Recent Inquiries (6 Months)", [0, 1, 2, 3, 4, 5], index=1)
loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000, max_value=200000, value=25000, step=2500)
employment_years = st.sidebar.slider("Employment Length (Years)", 0, 40, 5)

applicant_data = {
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "debt_to_income": debt_to_income,
    "delinquencies_2yrs": delinquencies_2yrs,
    "recent_inquiries": recent_inquiries,
    "loan_amount": loan_amount,
    "employment_years": employment_years
}

# Run Evaluation
results = agent.evaluate_applicant(applicant_data)

# Main Dashboard Layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("Risk Decision Status", results["decision"])

with col2:
    st.metric("Default Probability", f"{results['prob_default']*100:.1f}%")

with col3:
    st.metric("Risk Classification Tier", results["risk_tier"].split(" - ")[0])

st.divider()

# Tabbed detailed views
tab1, tab2, tab3 = st.tabs(["📊 SHAP Feature Impact", "🤖 AI Compliance & Explanation Report", "📜 Policy Guidelines"])

with tab1:
    st.subheader("Feature Impact Breakdown (SHAP Values)")
    st.caption("Positive SHAP values push default risk higher; negative values push default risk lower.")
    
    impacts = pd.DataFrame(results["shap_analysis"]["feature_impacts"])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#EF4444' if val > 0 else '#10B981' for val in impacts['shap_value']]
    
    sns.barplot(
        data=impacts,
        x='shap_value',
        y='feature',
        palette=colors,
        ax=ax
    )
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_title("SHAP Feature Attributions")
    ax.set_xlabel("Impact on Default Probability (SHAP Value)")
    ax.set_ylabel("Applicant Feature")
    st.pyplot(fig)

with tab2:
    st.markdown(results["markdown_report"])

with tab3:
    st.subheader("Underwriting Policy & Regulatory Documentation")
    summary = agent.knowledge_base.get_all_policy_summary()
    st.markdown(summary)
