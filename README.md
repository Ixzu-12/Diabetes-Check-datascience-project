# Diabetes-Check-datascience-project
# Diabetes Prediction using Random Forest & Bayesian Optimization

This project predicts whether a person has diabetes using the **PIMA Diabetes Dataset**.  
The model uses:

- Random Forest Classifier  
- Hyperparameter tuning using **Bayesian Optimization (Hyperopt)**  
- Cross-validation using Stratified K-Fold  
- Evaluation metrics such as Accuracy, F1-score, F-beta score, ROC-AUC  
- A simple **CLI-based prediction system** for user inputs  

---

## 📌 Features

### ✔ Data Preprocessing
- Replaced zero values in:
  - Glucose
  - Insulin
  - Pregnancies
  - SkinThickness  
  with mean values.

### ✔ Machine Learning
- Splitting data using `train_test_split`
- Hyperparameter tuning using **Hyperopt (TPE)**
- Training final Random Forest model with best discovered params

### ✔ Model Evaluation
Uses:
- Accuracy Score  
- Classification Report  
- ROC-AUC Score  
- F-beta Score  
- Stratified K-Fold Cross-Validation  

### ✔ User Input Prediction
The program accepts user inputs for:
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

Then predicts:

