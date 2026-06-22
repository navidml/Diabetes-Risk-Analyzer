#  Diabetes Risk Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

###  Machine Learning Pipeline for Early Diabetes Risk Prediction

*An ensemble-based predictive system that analyzes patient health indicators and estimates diabetes risk using advanced machine learning techniques.*

</div>

---

##  Overview

Diabetes is one of the most common chronic diseases worldwide. Early detection can significantly improve treatment outcomes and quality of life.

This project develops a complete **Machine Learning Pipeline** for diabetes risk prediction using patient medical data. The workflow includes:

-  Data preprocessing and exploration
-  Feature importance analysis
-  Correlation analysis and visualization
-  Multiple machine learning models
-  Hyperparameter optimization with GridSearchCV
-  Ensemble Stacking Architecture
-  Model serialization for deployment

---
**Web Application:**  
https://ml-diabetes-risk-analyzer.streamlit.app/

##  Objectives

- Predict whether a patient is likely to have diabetes.
- Identify the most influential health indicators.
- Compare multiple machine learning algorithms.
- Improve predictive performance through ensemble learning.
- Create a reusable trained model for future applications.

---

##  Dataset Features

The model utilizes several medical attributes commonly used in diabetes diagnosis:

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Patient age |
| Outcome | Diabetes diagnosis (Target Variable) |

---

##  Exploratory Data Analysis

### Feature Importance Analysis

The project performs feature ranking using tree-based methods to understand which variables contribute most to prediction performance.

### Correlation Heatmap

A correlation matrix is generated to investigate relationships among variables and detect potential multicollinearity.

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
```

---

## 🏗 Model Architecture

The project combines multiple machine learning algorithms into a powerful ensemble model.

### Base Models

#### 1️ Logistic Regression
- Balanced class weights
- Regularization tuning
- Optimized using GridSearchCV

#### 2️ Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Feature robustness

#### 3️ Gradient Boosting
- Sequential learning strategy
- Reduces prediction errors
- Improves overall accuracy

---

##  Hyperparameter Optimization

Each model is optimized using:

```python
GridSearchCV(
    estimator=model,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
```

Optimization Metric:

-  ROC-AUC Score
-  5-Fold Cross Validation
-  Parallel Processing

---

##  Stacking Ensemble

The final prediction system is built using a **Stacking Classifier**.

```text
                 ┌─────────────────┐
                 │ LogisticRegression │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ Random Forest   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ GradientBoosting│
                 └────────┬────────┘
                          │
                          ▼
                Meta Logistic Regression
                          │
                          ▼
                    Final Prediction
```

### Why Stacking?

✔ Combines strengths of multiple models

✔ Improves generalization

✔ Reduces overfitting

✔ Produces more stable predictions

---

##  Evaluation

The final model is evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

```python
nl.evaluate_classification_model(
    y_true=y_test,
    y_pred=y_pred
)
```

---

## 💾Model Export

The trained ensemble model is saved for deployment:

```python
joblib.dump(
    stack_model,
    "diabetes_prediction_model.pkl"
)
```

---

##  Tech Stack

| Category | Tools |
|-----------|--------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Model Optimization | GridSearchCV |
| Serialization | Joblib |

---


##  Getting Started

### Clone Repository

```bash
git clone https://github.com/your-username/Diabetes-Risk-Analyzer.git

cd Diabetes-Risk-Analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook
```

---

##  Future Improvements

- Deep Learning implementation
- Explainable AI (SHAP)
- Streamlit Web Application
- FastAPI Deployment
- Docker Containerization
- Real-time Prediction API

---

##  Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

##  Support

If you found this project useful:

 Star the repository

 Fork the project

 Share it with others

---

<div align="center">

###  Early Detection Saves Lives

**Built with Python, Machine Learning, and Data Science**

</div>
