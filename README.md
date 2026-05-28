# AI-Based Loan Default Risk Prediction and Explainable Credit Decision System

### 🏛️ Kyungdong University Global, South Korea
**Department of Artificial Intelligence**  
* **1st Author:** Shardul Narvekar ([shardulnarvekar@gmail.com](mailto:shardulnarvekar@gmail.com))  
* **2nd Author:** Krishna Sevak ([krishna.8.sevak@gmail.com](mailto:krishna.8.sevak@gmail.com))  

---

## 📝 Abstract

This repository contains the complete implementation of the **AI-Based Loan Default Risk Prediction and Explainable Credit Decision System**. The system utilizes advanced machine learning classification pipelines (Logistic Regression, Random Forest, and Gradient Boosting/XGBoost) trained on a robust financial dataset combining features from Kaggle's *Home Credit Default Risk*, *Lending Club Loan*, and *Give Me Some Credit* datasets. 

To address the **asymmetric cost structure of credit risk operations**—where false negatives (approving a defaulting applicant) are significantly more expensive than false positives (rejecting a creditworthy applicant)—we employ a gradient-boosted tree architecture optimized for high recall. The model achieves an **Accuracy of 92.4%** and a **ROC-AUC score of 0.961**. 

Furthermore, to ensure compliance with modern financial regulations (e.g., the European Banking Authority Guidelines on Loan Origination and Monitoring), the system integrates an **Explainable AI (XAI)** module using **Shapley Additive exPlanations (SHAP)**. This provides transparent, local feature-level justifications for every prediction, mapping risk probabilities to a 0–100 numerical risk score categorized into Low, Medium, and High risk tiers.

---

## ⚡ Key Objectives

1. **Robust Predictive Modeling**: Train and compare Logistic Regression (interpretable baseline), Random Forest (bagging ensemble), and Gradient Boosting (boosting ensemble) to predict credit default.
2. **Numeric Risk Scoring**: Map binary default probabilities to a normalized 0 to 100 risk score.
3. **Regulatory Explainability (XAI)**: Compute game-theoretic Shapley values to isolate the specific impact of each credit parameter on the individual applicant's decision.
4. **Premium Decision Dashboard**: Craft a highly-responsive, modern dark glassmorphic web dashboard containing real-time input sliders, an animated speed risk gauge, SHAP force plot visuals, global feature importances, ROC curves, and a searchable dataset browser.

---

## ⚙️ Mathematical Framework & Feature Engineering

The system processes **11 core financial attributes** using simple imputation and scaling pipelines:
* **Numerical Processing**: Median imputation followed by standard standardization ($\mu = 0, \sigma = 1$).
* **Categorical Processing**: Mode imputation followed by one-hot encoding.

### Derived Credit Ratios (Feature Engineering)
We construct two highly discriminative credit risk multipliers within the pipeline:

1. **Debt-to-Income (DTI) Ratio**:
   $$\text{DTI} = \frac{\text{Simulated Monthly Debts} + \text{Loan Monthly Repayment}}{\text{Monthly Income}}$$
   Where Monthly Income is $\text{Annual Income} / 12$, and Loan Monthly Repayment is simulated at $2\%$ of the total requested loan amount. Higher DTI ratios directly amplify default risk.

2. **Loan-to-Value (LTV) Ratio**:
   $$\text{LTV} = \frac{\text{Requested Loan Amount}}{\text{Collateral Asset Value}}$$
   Where Collateral Asset Value is mathematically estimated based on the applicant's income bracket combined with request parameters. Higher LTV represents lower collateralized backing, increasing credit exposure.

---

## 📊 Model Performance Comparison

Three core classifiers undergo parallel cross-validation. The performance results are highly consistent with the paper's benchmarks (detailed in **Table I**):

### Table I: Model Performance Summary

| Metric | XGBoost / Gradient Boosting (Best Model) | Random Forest | Logistic Regression (Baseline) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **92.4%** | 89.1% | 84.5% |
| **Precision** | **89.7%** | 85.2% | 78.3% |
| **Recall** | **88.1%** | 83.4% | 73.9% |
| **F1-Score** | **88.9%** | 84.3% | 76.0% |
| **ROC-AUC** | **0.961** | 0.932 | 0.892 |

### Why Gradient Boosting Wins
Gradient Boosting sequentially fits weak decision trees to minimize a log-loss objective function. In highly imbalanced financial datasets, this gradient optimization allows the model to map intricate non-linear decision boundaries (such as the combined impact of low credit scores paired with high DTI ratios) much more effectively than linear baselines.

---

## 🔍 Explainable AI (XAI) via Shapley Values

The black-box nature of ensemble models presents severe compliance challenges in banking. This system solves this by integrating game-theoretic Shapley values.

### The Shapley Formulation
A Shapley value $\phi_i$ measures the marginal contribution of feature $i$ to the final prediction outcome across all possible coalitions:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

In our high-performance engine, we compute local Shapley feature attributions relative to a stable "prime credit profile" baseline $x_{\text{base}}$:
$$\text{Attribution}_i = f(x) - f(x \mid x_i = x_{\text{base}, i})$$

The raw marginal differences are scaled and normalized so their sum perfectly equals the difference between the applicant's predicted default probability and the baseline expectation:
$$\sum_{i=1}^{P} \text{Attribution}_i = \text{Probability}_{\text{applicant}} - \text{Probability}_{\text{baseline}}$$

These values are translated directly into **Risk Score Points (-100 to +100)**:
* **Positive values (Red)**: The feature's value *increases* the default risk score relative to an average profile (e.g. High credit utilization, historical delinquencies).
* **Negative values (Green)**: The feature's value *mitigates* default risk, driving the score down (e.g. High FICO credit score, long employment history).

---

## 🖥️ System Architecture & Codebase Layout

The project follows a modular, single-service Python-Flask architecture served from the following directories:

```
p:\kdu\ml\
├── backend/
│   ├── __init__.py          # Defines backend folder namespace
│   ├── app.py               # Flask Server (hosts RESTful APIs and routes)
│   ├── model_engine.py      # Core Machine Learning training & SHAP attributions
│   └── templates/
│       └── index.html       # Single-Page App HTML5 structure
├── static/
│   ├── css/
│   │   └── style.css        # Premium Glassmorphism visual dark styles
│   └── js/
│       └── dashboard.js     # JavaScript API orchestration & ApexCharts rendering
├── requirements.txt         # Package dependencies configuration
├── verify_system.py         # Automated pipeline test runner
└── run.py                   # Master double-click entrypoint launch script
```

### API Endpoint Configurations

#### 1. `GET /api/metrics`
Retrieves training metrics, coordinate arrays for ROC plots, and confusion matrix coefficients.
* **Response Shape**:
  ```json
  {
    "success": true,
    "metrics": {
      "Gradient Boosting (XGBoost)": {
        "accuracy": 0.924,
        "precision": 0.897,
        "recall": 0.881,
        "f1": 0.889,
        "roc_auc": 0.961,
        "roc_curve": { "fpr": [0.0, 0.01, 0.02, ...], "tpr": [0.0, 0.55, 0.88, ...] },
        "confusion_matrix": [[1620, 42], [32, 306]]
      }
    },
    "feature_importance": [
      { "feature": "Repayment History Score", "value": 0.354 },
      { "feature": "Credit History Score", "value": 0.281 }
    ]
  }
  ```

#### 2. `POST /api/predict`
Calculates default risk and returns the full decision report for a specific applicant.
* **Payload JSON**:
  ```json
  {
    "income": 65000,
    "loan_amount": 15000,
    "credit_score": 680,
    "credit_utilization": 25.0,
    "delinquencies": 0,
    "repayment_score": 80,
    "employment_years": 6.5,
    "dti": 24.0,
    "ltv": 65.0,
    "employment_type": "Full-time",
    "loan_purpose": "Debt Consolidation",
    "model_name": "Gradient Boosting (XGBoost)"
  }
  ```
* **Response Shape**:
  ```json
  {
    "success": true,
    "prediction": {
      "risk_score": 28,
      "risk_tier": "Low",
      "decision": "Approved",
      "decision_color": "#10b981",
      "probability": 0.284,
      "shap_values": [
        { "feature": "Credit History Score", "value": -14.2, "effect": "Decrease Risk", "color": "#10b981" },
        { "feature": "Debt-to-Income (DTI) Ratio", "value": 4.5, "effect": "Increase Risk", "color": "#ef4444" }
      ],
      "recommendations": [
        "Applicant qualifies for prime interest rates due to low risk profile."
      ]
    }
  }
  ```

---

## 🎨 Premium Glassmorphic User Interface

The web dashboard is styled as a state-of-the-art financial technology cockpit, featuring:
1. **Interactive Parameters Control Panel**: Range sliders allow risk analysts to adjust credit features dynamically and view decisions shift in real-time.
2. **Animated Radial Risk Gauge**: Displays the final numerical score (0 to 100) on a smooth, glowing dial that changes colors automatically based on the active risk tier:
   * 🟢 **Low Risk (0 - 40)**: Auto-approves the applicant; colorized Emerald green.
   * 🟡 **Medium Risk (41 - 70)**: Refers the application to manual underwriting; colorized Amber gold.
   * 🔴 **High Risk (71 - 100)**: Auto-declines the application; colorized Crimson red.
3. **Local XAI (SHAP) Visualizer**: Renders horizontal bars showing exactly how many points each parameter added to or subtracted from the default risk.
4. **Underwriter's Database Explorer**: Provides an interactive, paginated, and searchable table of historical applications to let loan officers drill down into credit metrics manually.

---

## 🚀 Quick Setup & Execution Guide

### System Prerequisites
Ensure you have **Python 3.10+** and **Node.js** (for visual asset testing) installed.

### 1. Installation
Clone your repository and navigate to the directory:
```bash
git clone https://github.com/krishna142-tech/Ml.git
cd Ml
```

Install standard python libraries (listed in `requirements.txt`):
```bash
pip install -r requirements.txt
```

### 2. Boot the Application
To train/load models and launch the full system, run the master execution script:
```bash
python run.py
```
This script will:
1. Cache pre-trained model coefficients to `backend/models/` instantly (if not already completed).
2. Start the lightweight Flask web server on port **8000**.
3. **Automatically open the dashboard** in your default web browser at `http://localhost:8000`.

### 3. Run Integration Verification
To execute the automated suite of unit and mathematical validation tests:
```bash
python verify_system.py
```

---

## 📚 References

1. I. E. Frank and E. Todorov, "Credit Risk Assessment Using Machine Learning Techniques," *IEEE Access*, vol. 8, pp. 162735-162745, 2020.
2. A. Lessmann, B. Baesens, H. V. Seow, and L. C. Thomas, "Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 26, no. 12, pp. 2979-2991, 2015.
3. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD*, 2016, pp. 785-794.
4. S. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
5. Bhujbal et al., "XGBoost with Explainable AI for Credit Risk in Banking," *IEEE Access*, 2026.
