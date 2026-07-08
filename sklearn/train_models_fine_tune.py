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
from scipy.stats import randint, loguniform

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
#update on sklearn 1.11 ^^^

# other imports
import time
import joblib

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
y_train = pd.read_csv('y_train.csv').squeeze()  # squeeze превращает однострочный DataFrame в Series
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
y_logreg_pred = best_logreg.predict(X_test)

print("Logistic Regression")
print(f"Training time: {end_time - start_time:.4f} seconds")
print("Best parameters:", search.best_params_)

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
axes[0].text(0.01, 0.5, f"Accuracy: {acc:.4f}\nBest Params: {search.best_params_}\n\n{report}", fontsize=10, family='monospace')
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

#save results to text file
with open(result_dir / "logistic_regression_results.txt", "w") as f:
    f.write("Logistic Regression\n")
    f.write(f"Training time: {end_time - start_time:.4f} seconds\n")
    f.write(f"Best parameters: {search.best_params_}\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Confusion Matrix:\n{cnf_matrix}\n")

#save model
joblib.dump(best_logreg, models_dir / "logistic_regression_model.pkl")


## Decision Tree
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}
dec = DecisionTreeClassifier(random_state=89)
search = RandomizedSearchCV(
    dec, param_dist, n_iter=50, cv=5,
    scoring='f1_macro', random_state=89, n_jobs=-1
)
start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()
best_dec = search.best_estimator_
y_dec_pred = best_dec.predict(X_test)

print("Decision Tree")
print(f"Training time: {end_time - start_time:.4f} seconds")
print("Best parameters:", search.best_params_)

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
axes[0].text(0.01, 0.5, f"Accuracy: {acc:.4f}\nBest Params: {search.best_params_}\n\n{report}", fontsize=10, family='monospace')
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
try:
    dot_data = StringIO()
    export_graphviz(best_dec, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=X_train.columns, class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(str(result_dir / "decision_tree.png"))
except Exception as e:
    print(f"Skipping Decision Tree visualization. Graphviz might not be installed. Error: {e}")

#save results to text file
with open(result_dir / "decision_tree_results.txt", "w") as f:
    f.write("Decision Tree\n")
    f.write(f"Training time: {end_time - start_time:.4f} seconds\n")
    f.write(f"Best parameters: {search.best_params_}\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Confusion Matrix:\n{cnf_matrix}\n")

#save model
joblib.dump(best_dec, models_dir / "decision_tree_model.pkl")


## Random Forest
param_dist = {
    'n_estimators': randint(50, 200),
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}
rf = RandomForestClassifier(random_state=89)
search = RandomizedSearchCV(
    rf, param_dist, n_iter=30, cv=5,
    scoring='f1_macro', random_state=89, n_jobs=-1
)
start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()
best_rf = search.best_estimator_
y_rf_pred = best_rf.predict(X_test)

print("Random Forest")
print(f"Training time: {end_time - start_time:.4f} seconds")
print("Best parameters:", search.best_params_)

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
axes[0].text(0.01, 0.5, f"Accuracy: {acc:.4f}\nBest Params: {search.best_params_}\n\n{report}", fontsize=10, family='monospace')
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

#save results to text file
with open(result_dir / "random_forest_results.txt", "w") as f:
    f.write("Random Forest\n")
    f.write(f"Training time: {end_time - start_time:.4f} seconds\n")
    f.write(f"Best parameters: {search.best_params_}\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Confusion Matrix:\n{cnf_matrix}\n")

#save model
joblib.dump(best_rf, models_dir / "random_forest_model.pkl")

## Support Vector Machine (SVM)
param_dist = {
    'C': loguniform(1e-2, 1e2),
    'kernel': ['linear', 'rbf']
}
svm_model = svm.SVC(probability=True, random_state=89)
search = RandomizedSearchCV(
    svm_model, param_dist, n_iter=15, cv=3,
    scoring='f1_macro', random_state=89, n_jobs=-1
)
start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()
best_svm = search.best_estimator_
y_svm_pred = best_svm.predict(X_test)

print("Support Vector Machine (SVM)")
print(f"Training time: {end_time - start_time:.4f} seconds")
print("Best parameters:", search.best_params_)

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
axes[0].text(0.01, 0.5, f"Accuracy: {acc:.4f}\nBest Params: {search.best_params_}\n\n{report}", fontsize=10, family='monospace')
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

#save results to text file
with open(result_dir / "svm_results.txt", "w") as f:
    f.write("Support Vector Machine (SVM)\n")
    f.write(f"Training time: {end_time - start_time:.4f} seconds\n")
    f.write(f"Best parameters: {search.best_params_}\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Confusion Matrix:\n{cnf_matrix}\n")

#save model
joblib.dump(best_svm, models_dir / "svm_model.pkl")



## Neural Network MLP
param_dist = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (100,), (50, 50)],
    'alpha': loguniform(1e-5, 1e-1),
    'learning_rate_init': loguniform(1e-4, 1e-2)
}
mlp = MLPClassifier(max_iter=3000, random_state=89)
search = RandomizedSearchCV(
    mlp, param_dist, n_iter=15, cv=3,
    scoring='f1_macro', random_state=89, n_jobs=-1
)
start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()
best_mlp = search.best_estimator_
y_mlp_pred = best_mlp.predict(X_test)

print("Neural Network MLP")
print(f"Training time: {end_time - start_time:.4f} seconds")
print("Best parameters:", search.best_params_)

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
axes[0].text(0.01, 0.5, f"Accuracy: {acc:.4f}\nBest Params: {search.best_params_}\n\n{report}", fontsize=10, family='monospace')
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

#save results to text file
with open(result_dir / "mlp_results.txt", "w") as f:
    f.write("Neural Network MLP\n")
    f.write(f"Training time: {end_time - start_time:.4f} seconds\n")
    f.write(f"Best parameters: {search.best_params_}\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Confusion Matrix:\n{cnf_matrix}\n")

#save model
joblib.dump(best_mlp, models_dir / "mlp_model.pkl")

print("All models trained, evaluated, and saved successfully.")