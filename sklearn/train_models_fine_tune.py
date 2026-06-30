# Data handling
import pandas as pd
import numpy as np
from pathlib import Path

# Data preprocessing
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    label_binarize
)

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# Metrics and evaluation
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Decision tree visualization
from sklearn.tree import export_graphviz
from IPython.display import Image
from io import StringIO
import pydotplus

# Random search utilities
from scipy.stats import randint

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
#update on sklearn 1.11 ^^^

# other imports
import time
import joblib

# fine tune
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# Paths and files
models_dir = Path("models_tuned")
models_dir.mkdir(exist_ok=True)
result_dir = Path("results_tuned")
result_dir.mkdir(exist_ok=True)

# Target variable
target_column = "Performance_Category"
class_names = ['Low', 'Medium', 'High']

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()


## Logistic Regression
param_dist = {
    'C': loguniform(1e-3, 1e2),
    'solver': ['saga'],
    'penalty': ['elasticnet'],
    'l1_ratio': [0.0, 0.5, 1.0],
    'class_weight': [None, 'balanced']
}
logreg = LogisticRegression(random_state=89, max_iter=2000)
search = RandomizedSearchCV(
    logreg, param_dist, n_iter=50, cv=5,
    scoring='f1_macro', random_state=89, n_jobs=-1
)
start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()
best_logreg = search.best_estimator_
print(search.best_params_)
y_logreg_pred = best_logreg.predict(X_test)

print("Logistic Regression")
print(f"Training time: {end_time - start_time:.4f} seconds")

#accuracy
acc = accuracy_score(y_test, y_logreg_pred)
print("Accuracy:", acc)
#Precision, Recall, F1-score
report = metrics.classification_report(y_test, y_logreg_pred)
print("Classification Report:\n", report)
#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_logreg_pred)
print("Confusion Matrix:\n", cnf_matrix)

#visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].text(0.01, 0.5, f"Accuracy: {acc:.4f}\n\n{report}", fontsize=10, family='monospace')
axes[0].axis('off')
tick_marks = np.arange(len(class_names))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].xaxis.set_label_position("top")
axes[1].set_title('Confusion matrix', y=1.1)
axes[1].set_ylabel('Actual label')
axes[1].set_xlabel('Predicted label')
plt.tight_layout()
plt.savefig(result_dir / "logistic_regression_results.png", dpi=150, bbox_inches="tight")
plt.clf()

#save model
joblib.dump(best_logreg, models_dir / "logistic_regression_model.pkl")


print("All models trained, evaluated, and saved successfully.")