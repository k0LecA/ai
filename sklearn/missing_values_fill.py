import pandas as pd
import numpy as np

df = pd.read_csv('Student_Productivity_Dataset.csv')

print(df.isnull().sum())

numeric_cols = [
    'Age',
    'Study_Hours_Per_Day',
    'Sleep_Hours_Per_Night',
    'Screen_Time_Hours',
    'Social_Media_Hours',
    'Attendance_Percentage',
    'Assignments_Completed',
    'Class_Participation_Score',
    'Physical_Activity_Hours_Per_Week',
    'Stress_Level',
    'Motivation_Level',
    'Extracurricular_Involvement',
    'AI_Tool_Usage_Hours_Per_Week',
    'Previous_Semester_GPA'
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = [
    'Gender',
    'Internet_Quality',
    'Part_Time_Job'
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv('filled_data.csv', index=False)
print("Missing values filled and saved to 'filled_data.csv'.")
