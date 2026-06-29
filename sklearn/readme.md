1 filled missing data

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

for numeric values used median()
for categorical values user mode()

2 normalize data

df['Gender'] = df['Gender'].map({
    'Male': 0,
    'Female': 1
})

maped category columns: gender,internet_quality,part_time_job, performance_category

scaler = StandardScaler()

numeric_cols = ['Age', 'Study_Hours_Per_Day', 'Sleep_Hours_Per_Night', 'Screen_Time_Hours', 'Social_Media_Hours', 'Attendance_Percentage', 'Assignments_Completed', 'Class_Participation_Score', 'Physical_Activity_Hours_Per_Week', 'Stress_Level', 'Motivation_Level', 'Extracurricular_Involvement', 'AI_Tool_Usage_Hours_Per_Week', 'Previous_Semester_GPA', 'Internet_Quality', 'Performance_Category']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

normalized numeric columns with standart scaler

3 model training

logistic regression
decision tree
random forest
support vector machine
neural network (MLP)