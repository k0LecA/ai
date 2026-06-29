import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('filled_data.csv').drop(columns=['Student_ID'])

df['Gender'] = df['Gender'].map({
    'Male': 0,
    'Female': 1
})

df['Part_Time_Job'] = df['Part_Time_Job'].map({
    'No': 0,
    'Yes': 1
})

df['Internet_Quality'] = df['Internet_Quality'].map({
    'Poor': 0,
    'Average': 1,
    'Good': 2
})
df['Performance_Category'] = df['Performance_Category'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

scaler = StandardScaler()

numeric_cols = ['Age', 'Study_Hours_Per_Day', 'Sleep_Hours_Per_Night', 'Screen_Time_Hours', 'Social_Media_Hours', 'Attendance_Percentage', 'Assignments_Completed', 'Class_Participation_Score', 'Physical_Activity_Hours_Per_Week', 'Stress_Level', 'Motivation_Level', 'Extracurricular_Involvement', 'AI_Tool_Usage_Hours_Per_Week', 'Previous_Semester_GPA', 'Internet_Quality']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#df[['Age']] = scaler.fit_transform(df[['Age']])
#df[['Study_Hours_Per_Day']] = scaler.fit_transform(df[['Study_Hours_Per_Day']])
#df[['Sleep_Hours_Per_Night']] = scaler.fit_transform(df[['Sleep_Hours_Per_Night']])
#df[['Screen_Time_Hours']] = scaler.fit_transform(df[['Screen_Time_Hours']])
#df[['Social_Media_Hours']] = scaler.fit_transform(df[['Social_Media_Hours']])
#df[['Attendance_Percentage']] = scaler.fit_transform(df[['Attendance_Percentage']])
#df[['Assignments_Completed']] = scaler.fit_transform(df[['Assignments_Completed']])
#df[['Class_Participation_Score']] = scaler.fit_transform(df[['Class_Participation_Score']])
#df[['Physical_Activity_Hours_Per_Week']] = scaler.fit_transform(df[['Physical_Activity_Hours_Per_Week']])
#df[['Stress_Level']] = scaler.fit_transform(df[['Stress_Level']])
#df[['Motivation_Level']] = scaler.fit_transform(df[['Motivation_Level']])
#df[['Extracurricular_Involvement']] = scaler.fit_transform(df[['Extracurricular_Involvement']])
#df[['AI_Tool_Usage_Hours_Per_Week']] = scaler.fit_transform(df[['AI_Tool_Usage_Hours_Per_Week']])
#df[['Previous_Semester_GPA']] = scaler.fit_transform(df[['Previous_Semester_GPA']])

df.to_csv('normalized_data.csv', index=False)
print("Data normalization completed and saved to 'normalized_data.csv'.")