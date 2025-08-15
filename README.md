# Credit Score Classification Using ANN  

## Overview  
This project develops a robust **Artificial Neural Network (ANN)** model to classify customers into three credit score categories — **Good**, **Standard**, and **Poor** — based on demographic, financial, and behavioral attributes.  
A **Streamlit dashboard** enables interactive hyperparameter tuning and performance visualization, making it a practical tool for financial institutions to support risk assessment and lending decisions.  

🔗 **Live App:** [Credit Score ANN Dashboard](https://anndashboard-iuhhup3cgdfdais4nbvaq2.streamlit.app/)  
💡 **Tip:** Use the buttons in the filter pane for hyperparameter tuning.  

---

## Features  
- Comprehensive **data preprocessing**: Missing value handling, encoding, and scaling.  
- **Feature engineering** for detailed loan type analysis.  
- Class imbalance resolution using **SMOTE**.  
- ANN model optimized through **iterative architecture tuning**.  
- Interactive **Streamlit dashboard** for model experimentation.  
- Business-oriented insights for **loan approvals, interest rate setting, and risk management**.  

---

## Dataset  
- **Rows:** 100,000  
- **Columns:** 28 (Demographic, financial, and credit history details)  
- **Target Variable:** Credit_Score (`Good`, `Standard`, `Poor`)  

**Key Features:**  
- Demographics (Age, Occupation, Annual Income)  
- Financial behavior (Outstanding Debt, Credit Utilization Ratio, Payment Behaviour)  
- Loan details (Type, Number, Delay from Due Date)  
- Credit history (Length, Inquiries, Credit Mix)  

---

## Key Insights  
- Higher annual income and fewer delayed payments strongly correlate with **better credit scores**.  
- **Loan type distribution** reveals patterns linked to creditworthiness.  
- Multiple credit inquiries are associated with **higher financial risk**.  
- "Good" credit customers have **fewer loans, better payment habits**, and receive more favorable lending terms.  

---

## Technologies Used  
- **Python**  
- **TensorFlow/Keras**  
- **Pandas, NumPy**  
- **Matplotlib, Seaborn**  
- **scikit-learn**  
- **Streamlit**  

---

## Model Performance & Business Applications  
- ANN-7 with SMOTE achieved the **best balance between accuracy and recall**.  
- Predicts customer credit scores to support:  
  - **Loan approvals** with risk assessment  
  - **Interest rate personalization**  
  - **Customer segmentation** for targeted offerings  
- Enables **automated credit scoring** to improve decision speed and consistency.  

---

## Managerial Recommendations  
- Prioritize customers with high income and low payment delays.  
- Closely monitor loan applicants with multiple recent credit inquiries.  
- Adjust lending criteria for customers with “Standard” or “Poor” scores.  
- Continuously retrain the model with new market data for adaptability.  
