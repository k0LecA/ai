import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

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
#cnf_matrix

