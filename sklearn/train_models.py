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

# Paths and files
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
result_dir = Path("results")
result_dir.mkdir(exist_ok=True)

# Target variable
target_column = "Performance_Category"
class_names = ['Low', 'Medium', 'High']

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # squeeze превращает однострочный DataFrame в Series
y_test = pd.read_csv('y_test.csv').squeeze()


## Logistic Regression
logreg = LogisticRegression(random_state=89, max_iter=1000)
start_time = time.time()
logreg.fit(X_train, y_train)
end_time = time.time()
y_logreg_pred = logreg.predict(X_test)

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
joblib.dump(logreg, models_dir / "logistic_regression_model.pkl")


## Decision Tree
dec = DecisionTreeClassifier(random_state=89, max_depth=5)
start_time = time.time()
dec.fit(X_train, y_train)
end_time = time.time()
y_dec_pred = dec.predict(X_test)

print("Decision Tree")
print(f"Training time: {end_time - start_time:.4f} seconds")

#accuracy
acc = accuracy_score(y_test, y_dec_pred)
print("Decision Tree Accuracy:", acc)
#Precision, Recall, F1-score
report = metrics.classification_report(y_test, y_dec_pred)
print("Classification Report:\n", report)
#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_dec_pred)
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
plt.savefig(result_dir / "decision_tree_results.png", dpi=150, bbox_inches="tight")
plt.clf()

#visualize the decision tree
dot_data = StringIO()
export_graphviz(dec, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=X_train.columns, class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(str(result_dir / "decision_tree.png"))

#save model
joblib.dump(dec, models_dir / "decision_tree_model.pkl")


## Random Forest
rf=RandomForestClassifier(random_state=89, max_depth=5)
start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()
y_rf_pred = rf.predict(X_test)

print("Random Forest")
print(f"Training time: {end_time - start_time:.4f} seconds")

#accuracy
acc = accuracy_score(y_test, y_rf_pred)
print("Random Forest Accuracy:", acc)
#Precision, Recall, F1-score
report = metrics.classification_report(y_test, y_rf_pred)
print("Classification Report:\n", report)
#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_rf_pred)
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
plt.savefig(result_dir / "random_forest_results.png", dpi=150, bbox_inches="tight")
plt.clf()

#save model
joblib.dump(rf, models_dir / "random_forest_model.pkl")

## Support Vector Machine (SVM)
svm_model = svm.SVC(kernel='linear', probability=True, random_state=89)
start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()
y_svm_pred = svm_model.predict(X_test)

print("Support Vector Machine (SVM)")
print(f"Training time: {end_time - start_time:.4f} seconds")

#accuracy
acc = accuracy_score(y_test, y_svm_pred)
print("Support Vector Machine (SVM) Accuracy:", acc)
#Precision, Recall, F1-score
report = metrics.classification_report(y_test, y_svm_pred)
print("Classification Report:\n", report)
#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_svm_pred)
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
plt.savefig(result_dir / "svm_results.png", dpi=150, bbox_inches="tight")
plt.clf()

#save model
joblib.dump(svm_model, models_dir / "svm_model.pkl")



## Neural Network MLP

mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, random_state=89)
start_time = time.time()
mlp.fit(X_train, y_train)
end_time = time.time()
y_mlp_pred = mlp.predict(X_test)

print("Neural Network MLP")
print(f"Training time: {end_time - start_time:.4f} seconds")

#accuracy
acc = accuracy_score(y_test, y_mlp_pred)
print("Neural Network MLP Accuracy:", acc)
#Precision, Recall, F1-score
report = metrics.classification_report(y_test, y_mlp_pred)
print("Classification Report:\n", report)
#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_mlp_pred)
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
plt.savefig(result_dir / "mlp_results.png", dpi=150, bbox_inches="tight")
plt.clf()

#save model
joblib.dump(mlp, models_dir / "mlp_model.pkl")

print("All models trained, evaluated, and saved successfully.")