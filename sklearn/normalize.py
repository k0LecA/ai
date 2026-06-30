import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

df = pd.read_csv('filled_data.csv').drop(columns=['Student_ID'])

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['Part_Time_Job'] = df['Part_Time_Job'].map({'No': 0, 'Yes': 1})
df['Internet_Quality'] = df['Internet_Quality'].map({'Poor': 0, 'Average': 1, 'Good': 2})
df['Performance_Category'] = df['Performance_Category'].map({'Low': 0, 'Medium': 1, 'High': 2})

target_column = "Performance_Category"

X = df.drop(columns=[target_column, 'Productivity_Score'])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=89, stratify=y
)

# Скейлим числовые признаки: fit только на train
numeric_cols = ['Age', 'Study_Hours_Per_Day', 'Sleep_Hours_Per_Night', 'Screen_Time_Hours',
                 'Social_Media_Hours', 'Attendance_Percentage', 'Assignments_Completed',
                 'Class_Participation_Score', 'Physical_Activity_Hours_Per_Week', 'Stress_Level',
                 'Motivation_Level', 'Extracurricular_Involvement', 'AI_Tool_Usage_Hours_Per_Week',
                 'Previous_Semester_GPA', 'Internet_Quality']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Сохраняем готовые куски
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

joblib.dump(scaler, models_dir / 'scaler.pkl')

print("Normalization + split completed. Train/test files saved.")