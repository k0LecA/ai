# Email Spam Classification with TF-IDF & Logistic Regression

This folder contains a simple machine learning program to classify emails as **Ham** (legitimate) or **Spam** using TF-IDF feature extraction and a Logistic Regression classifier.

---

## Files

**Data**
- `spam_ham_dataset.csv` — the email dataset (5,171 samples). Source: [Kaggle Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).

**Code**
- `classify.py` — main pipeline script: loads data, splits it, runs TF-IDF vectorization, trains Logistic Regression, evaluates results, and saves plots/metrics.

**Outputs**
- `results/results.txt` — model configuration, performance metrics, and results analysis.
- `results/confusion_matrix.png` — confusion matrix heatmap image.
- `models/model.pkl` & `models/vectorizer.pkl` — saved trained Logistic Regression model and TF-IDF vectorizer.

---

## Results Summary

- **Accuracy**: 98.61%
- **Training Time**: ~0.04 seconds
- **Confusion Matrix**:
  - True Ham: 908 classified correctly (10 misclassified as Spam)
  - True Spam: 367 classified correctly (8 misclassified as Ham)

For a detailed analysis, see `results/results.txt`.

---

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## Running the Program

```bash
venv/bin/python3 classify.py
```