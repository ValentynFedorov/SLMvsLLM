# README

##  Objective of the Experiment

The purpose of this study is to compare different machine learning algorithms for predicting student academic outcomes (classes: **Graduate**, **Dropout**, **Enrolled**). The methods CatBoost, XGBoost, and Random Forest were used. Additionally, statistical tests were conducted to analyze differences between data distributions.

---

##  Experiment Workflow

### 1. Environment Setup
Python environment with the following libraries:
- `pandas`, `numpy` — data processing
- `scikit-learn` — train/test split, evaluation metrics
- `catboost`, `xgboost`, `sklearn.ensemble.RandomForestClassifier` — models
- `matplotlib`, `seaborn` — visualization
- `scipy.stats` — statistical tests

### 2. Data Preparation
- Dataset `data.csv` was loaded.
- Target variable `Target` was encoded as follows:
  - Graduate → 0  
  - Dropout → 1  
  - Enrolled → 2  

### 3. Train/Test Split
- `train_test_split` was used (80% training, 20% testing).

### 4. Models and Parameters
- **CatBoost**: 1000 iterations, depth=8, learning_rate=0.01  
- **XGBoost**: 500 trees, learning_rate=0.01, max_depth=8, subsample=0.8  
- **Random Forest**: 500 trees, max_depth=12, bootstrap=True  

### 5. Evaluation Metrics
The following metrics were used to evaluate model performance:
- **Accuracy** — overall classification correctness.  
- **Precision** — proportion of correct positive predictions among all predicted positives.  
- **Recall** — proportion of correct positive predictions among all actual positives.  
- **F1-score** — harmonic mean of Precision and Recall.  
- **ROC-AUC** — ability to distinguish between classes.  

### 6. Statistical Tests
- **KS-test (Kolmogorov-Smirnov)**: tests whether two distributions differ significantly.  
- **Wasserstein distance**: measures the difference between two distributions.  
- **Entropy**: information-theoretic measure of uncertainty.  

---

##  Results

| Model           | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-----------------|----------|-----------|--------|----------|---------|
| CatBoost        | …        | …         | …      | …        | …       |
| XGBoost         | …        | …         | …      | …        | …       |
| Random Forest   | …        | …         | …      | …        | …       |

 Actual metrics should be inserted here after running the notebook.

---

##  Visualizations

- ROC curves for all models  
- Confusion Matrix  
- Metric comparison (bar charts)  
- Feature Importance  

 Placeholders for screenshots:

```md
![ROC Curve](images/roc_curve.png)
![Confusion Matrix](images/confusion_matrix.png)
![Feature Importance](images/feature_importance.png)
