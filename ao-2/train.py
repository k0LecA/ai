import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')

X = []
y = []

for _, row in df.iterrows():
    img = cv2.imread(row['path'])
    img = cv2.resize(img, (64, 64))

    X.append(img.flatten())
    y.append(row['label'])

X = np.array(X)
X = X / 255.0

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=89
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("OOB Score:", model.oob_score_)