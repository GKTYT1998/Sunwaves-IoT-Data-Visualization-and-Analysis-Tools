# Sunwaves-IoT-Data-Visualization-and-Analysis-Tools
# Hypertension Prediction Using Machine Learning

A machine learning project for predicting hypertension risk using clinical and lifestyle features, developed as part of a Medical Supervision Center (MSC) initiative.

## Overview

This project applies multiple machine learning algorithms to predict hypertension risk and uses SHAP analysis for model interpretability.

## Dataset

- **Source**: [Kaggle - Hypertension Dataset](https://www.kaggle.com/datasets/sumedh1507/hypertension-dataset/data)
- **Size**: 1,985 patient records
- **Features**: Age, BMI, Salt_Intake, Stress_Score, Sleep_Duration, BP_History, Smoking_Status, Exercise_Level, Medication, Family_History
- **Target**: Has_Hypertension (Binary: 1 = Hypertensive, 0 = Normotensive)

## Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- MLP Classifier

## Results

| Model | Accuracy |
|-------|----------|
| AdaBoost | 100.00% |
| XGBoost | 99.50% |
| Gradient Boosting | 98.74% |
| Random Forest | 96.73% |
| Decision Tree | 96.22% |
| SVM | 89.92% |
| MLP Classifier | 89.67% |
| KNN | 86.90% |
| Logistic Regression | 82.62% |

## SHAP Feature Importance (Top 5)

1. BP_History (0.068)
2. Family_History (0.044)
3. Age (0.036)
4. Smoking_Status (0.031)
5. Stress_Score (0.025)

## Installation
```bash
git clone https://github.com/yourusername/hypertension-prediction.git
cd hypertension-prediction
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

## Usage
```bash
jupyter notebook hypertension_ML_sorted_result.ipynb
```

## Requirements

- Python 3.13+
- pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn

## References

- Sumedh1507. (2024). *Hypertension Dataset*. Kaggle. https://www.kaggle.com/datasets/sumedh1507/hypertension-dataset/data
- Sumedh1507. (2024). *Hypertension Analysis and Prediction*. Kaggle. https://www.kaggle.com/code/sumedh1507/hypertension-analysis-and-prediction

## License

MIT License
