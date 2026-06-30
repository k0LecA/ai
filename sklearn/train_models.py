import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

df = pd.read_csv('normalized_data.csv')

X=df.drop(columns=['Performance_Category', 'Productivity_Score'])
y=df[['Performance_Category','Productivity_Score']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=89)



##logistic regression
logreg = LogisticRegression(random_state=89)

logreg.fit(X_train, y_train['Performance_Category'])

y_pred = logreg.predict(X_test)

#confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test['Performance_Category'], y_pred)
#print(cnf_matrix)
#cnf_matrix
print("Logistic Regression Accuracy:", accuracy_score(y_test['Performance_Category'], y_pred))


class_names = ['Low', 'Medium', 'High']
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g',xticklabels=class_names, yticklabels=class_names)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#plt.savefig("rez.png", dpi=150, bbox_inches="tight")
plt.savefig("cnf_logreg.png", dpi=150, bbox_inches="tight")
#Text(0.5,257.44,'Predicted label');

from sklearn.metrics import classification_report
target_names = ['Low', 'Medium', 'High']
#print(classification_report(y_test['Performance_Category'], y_pred, target_names=target_names))
plt.clf()
report = metrics.classification_report(y_test['Performance_Category'], y_pred)

plt.figure(figsize=(6, 4))
plt.text(0.01, 0.5, report, fontsize=10, family='monospace')
plt.axis('off')
plt.savefig("class_report_logreg.png", dpi=150, bbox_inches="tight")

from sklearn.preprocessing import label_binarize
plt.clf()
y_test_bin = label_binarize(y_test['Performance_Category'], classes=[0,1,2])
y_pred_proba = logreg.predict_proba(X_test)

for i in range(3):
    fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    auc = metrics.roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"class {i}, auc={auc:.2f}")

plt.legend(loc=4)
plt.savefig("roc_logreg.png", dpi=150, bbox_inches="tight")
plt.clf()

##Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=89, max_depth=5)
clf.fit(X_train, y_train['Performance_Category'])

y_pred_tree = clf.predict(X_test)


print("Decision Tree Classifier Accuracy:", accuracy_score(y_test['Performance_Category'], y_pred_tree))

#visualize the decision tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=X.columns, class_names=['Low', 'Medium', 'High'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("decision_tree.png")
#Image(graph.create_png())

plt.clf()
class_names=['Low', 'Medium', 'High'] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
cnf_matrix = metrics.confusion_matrix(y_test['Performance_Category'], y_pred_tree)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g',xticklabels=class_names, yticklabels=class_names)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig("cnf_decision_tree.png", dpi=150, bbox_inches="tight")


##Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image


rf=RandomForestClassifier(random_state=89, max_depth=5)
rf.fit(X_train, y_train['Performance_Category'])

y_pred_rf = rf.predict(X_test)

accuracy = accuracy_score(y_test['Performance_Category'], y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy)



##Support Vector Machine (SVM)
from sklearn import svm

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
#update on sklearn 1.11

svm_model = svm.SVC(kernel='linear', probability=True, random_state=89)
svm_model.fit(X_train, y_train['Performance_Category'])
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test['Performance_Category'], y_pred_svm)
print("Support Vector Machine (SVM) Accuracy:", accuracy_svm)

## Neural Network MLP
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, random_state=89)
mlp.fit(X_train, y_train['Performance_Category'])
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test['Performance_Category'], y_pred_mlp)
print("Neural Network MLP Accuracy:", accuracy_mlp)