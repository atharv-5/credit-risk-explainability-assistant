# Credit Risk Explainability (XAI & AI Agent System)

An end-to-end Explainable AI (XAI) and AI Agent system for transparent, compliant, and actionable Credit Risk Assessment.

## Features

- **Machine Learning Risk Model**: Predicts credit default probability using XGBoost trained on financial & credit history features.
- **SHAP Feature Attribution**: Generates feature-level explanations (waterfall plots & feature impacts) for individual loan decisions.
- **Policy Knowledge Base**: Indexes credit underwriting guidelines and regulatory compliance constraints.
- **AI Explanation Agent**: Synthesizes prediction scores, SHAP attributions, and policy rules to generate human-readable credit decision reports and remediation advice.
- **Interactive Dashboard**: Streamlit interface featuring dynamic applicant input, real-time risk scoring, SHAP visualizer, and "What-If" scenario simulator.

## Project Structure

```
credit-risk-explainability/
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Processed feature matrices
├── docs/
│   └── policy/              # Credit policy & compliance rules
├── models/                  # Trained ML model & scaler artifacts
├── notebooks/
│   └── 01_model_training.ipynb # Model training & evaluation
├── src/
│   ├── __init__.py
│   ├── explain.py           # SHAP explanation engine & counterfactual generator
│   ├── knowledge_base.py    # Policy RAG & rule search engine
│   └── agent.py             # Decision report synthesis agent
├── app.py                   # Streamlit interactive application
├── requirements.txt         # Python dependencies
└── README.md
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   Run the training notebook `notebooks/01_model_training.ipynb` or execute the training pipeline to save model artifacts into `models/`.

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```
