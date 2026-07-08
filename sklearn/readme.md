# Student Productivity Classification Project

This project focuses on predicting student academic performance categories (`Low`, `Medium`, `High`) based on various demographic, behavioral, and academic features from the [Student Productivity Dataset](https://www.kaggle.com/datasets/velvetcrystal/student-productivity-dataset).

---

## Project Structure

* **Preprocessing & Imputation**:
  * [missing_values_fill.py](file:///home/shark/Projects/ai/sklearn/missing_values_fill.py): Imputes missing numeric values with their median and categorical values with their mode.
  * [normalize.py](file:///home/shark/Projects/ai/sklearn/normalize.py): One-hot encodes nominal features (e.g. `Gender`), maps ordinal features, performs a stratified train/test split, scales features with `StandardScaler` (fit on train only), and saves data splits.
* **Model Training & Evaluation**:
  * [train_models.py](file:///home/shark/Projects/ai/sklearn/train_models.py): Trains baseline models (Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network MLP), generates performance report text files, and saves confusion matrix plots.
  * [train_models_fine_tune.py](file:///home/shark/Projects/ai/sklearn/train_models_fine_tune.py): Fine-tunes all models using `RandomizedSearchCV` to optimize F1-macro score.
* **Exploratory Data Analysis**:
  * [correlation.py](file:///home/shark/Projects/ai/sklearn/correlation.py): Plots a correlation heatmap of the imputed numeric features.
* **Output Folders**:
  * `results/` & `results_tuned/`: Contain classification reports, confusion matrices, and decision tree graphs.
  * `models/` & `models_tuned/`: Contain serialized models (`.pkl`) and the pre-fitted scaler.

---

## Data Pipeline & Key Modifications

1. **Target Leakage Prevention**: `Productivity_Score` has been dropped from the feature matrix $X$. The target `Performance_Category` is a deterministic discretization of `Productivity_Score`. Leaving it in the features leads to trivial $100\%$ accuracy, masking the model's actual utility.
2. **Data Leakage Fix**: Feature scaling (`StandardScaler`) is fitted *only* on the training split to prevent test-set characteristics from leaking into the training pipeline.
3. **One-Hot Encoding**: `Gender` (`Male`, `Female`, `Other`) is encoded using One-Hot encoding (dropping the `Female` reference category) instead of integer label encoding. This prevents distance/linear-based models from assuming an artificial ordering constraint.

---

## Model Performance Summary

The performance of the models on the test set ($2,500$ samples) is summarized below. Detailed metric breakdowns are saved in [all_results.txt](file:///home/shark/Projects/ai/sklearn/all_results.txt).

### Baseline Models (Default Parameters)
* **Logistic Regression**: **72.40%** Accuracy (F1-macro: 0.71)
* **Support Vector Machine (SVM)**: **72.32%** Accuracy (F1-macro: 0.71)
* **Neural Network (MLP)**: **69.04%** Accuracy (F1-macro: 0.69)
* **Random Forest** (max_depth=5): **58.60%** Accuracy (F1-macro: 0.46)
* **Decision Tree** (max_depth=5): **54.72%** Accuracy (F1-macro: 0.48)

### Fine-Tuned Models (Randomized Search CV)
* **Support Vector Machine (SVM)**: **74.16%** Accuracy (F1-macro: 0.73) &mdash; **Best Model**
* **Neural Network (MLP)**: **73.64%** Accuracy (F1-macro: 0.73)
* **Logistic Regression**: **72.40%** Accuracy (F1-macro: 0.71)
* **Random Forest**: **68.92%** Accuracy (F1-macro: 0.66)
* **Decision Tree**: **57.40%** Accuracy (F1-macro: 0.56)

---

## Quick Start

### 1. Set Up Environment
Ensure the virtual environment is set up and required packages are installed:

> [!NOTE]
> System-level **Graphviz** is required to generate the Decision Tree structure diagrams. On Debian/Ubuntu, you can install it using `sudo apt install graphviz`. If Graphviz is missing, tree visualization steps will be skipped gracefully.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # (or install pandas, numpy, scikit-learn, matplotlib, seaborn, pydotplus, joblib)
```

### 2. Preprocess the Data
```bash
python3 missing_values_fill.py
python3 normalize.py
```

### 3. Train Models
To train the baseline models:
```bash
python3 train_models.py
```
To run hyperparameter tuning:
```bash
python3 train_models_fine_tune.py
```