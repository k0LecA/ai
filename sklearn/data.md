# Student Productivity Dataset Info

The dataset used in this project is the **Student Productivity Dataset**, which compiles demographic, behavioral, and academic indicators for $10,000$ students.

* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/velvetcrystal/student-productivity-dataset)
* **Shape**: $10,000$ rows, $20$ columns.

---

## Dataset Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Student_ID** | `int64` | Unique student identifier (dropped during preprocessing). |
| **Age** | `float64` | Age of the student (18 to 30). |
| **Gender** | `object` | Nominally encoded gender (`Male`, `Female`, `Other`). |
| **Study_Hours_Per_Day** | `float64` | Average daily study hours (0 to 10). |
| **Sleep_Hours_Per_Night** | `float64` | Average nightly sleep hours (4 to 10). |
| **Screen_Time_Hours** | `float64` | Average daily leisure screen time (0 to 12). |
| **Social_Media_Hours** | `float64` | Average daily hours spent on social media (0 to 8). |
| **Attendance_Percentage** | `float64` | Class attendance rate (0% to 100%). |
| **Assignments_Completed** | `float64` | Total number of assignments finished (0 to 10). |
| **Class_Participation_Score** | `float64` | Score evaluating engagement during lectures (0 to 10). |
| **Physical_Activity_Hours_Per_Week** | `float64` | Weekly exercise hours (0 to 15). |
| **Stress_Level** | `float64` | Self-reported stress score (0 to 10). |
| **Motivation_Level** | `float64` | Self-reported motivation score (0 to 10). |
| **Internet_Quality** | `object` | Quality level of connection (`Poor`, `Average`, `Good`). |
| **Part_Time_Job** | `object` | Employment status (`No`, `Yes`). |
| **Extracurricular_Involvement** | `float64` | Hours per week in extracurricular activities. |
| **AI_Tool_Usage_Hours_Per_Week** | `float64` | Time spent using AI assistance tools. |
| **Previous_Semester_GPA** | `float64` | Academic score from prior semester (0 to 4). |
| **Productivity_Score** | `float64` | Continuous index scoring productivity (0 to 100). |
| **Performance_Category** | `object` | Target category (`Low`, `Medium`, `High`). |

---

## Missing Value Details

The raw dataset contains missing values across multiple behavioral features:

* **Age**: 155 missing
* **Gender**: 119 missing
* **Study_Hours_Per_Day**: 165 missing
* **Sleep_Hours_Per_Night**: 106 missing
* **Screen_Time_Hours**: 132 missing
* **Social_Media_Hours**: 80 missing
* **Attendance_Percentage**: 122 missing
* **Assignments_Completed**: 118 missing
* **Class_Participation_Score**: 185 missing
* **Physical_Activity_Hours_Per_Week**: 137 missing
* **Stress_Level**: 154 missing
* **Motivation_Level**: 175 missing
* **Internet_Quality**: 149 missing
* **Part_Time_Job**: 126 missing
* **Extracurricular_Involvement**: 178 missing
* **AI_Tool_Usage_Hours_Per_Week**: 140 missing
* **Previous_Semester_GPA**: 161 missing
* **Productivity_Score** / **Performance_Category**: 0 missing

### Imputation Policy
* **Numeric Features**: Missing fields are imputed using the column **median** to minimize skewness from extreme outliers.
* **Categorical Features**: Missing fields are imputed using the column **mode** (most frequent class).

---

## Preprocessing & Encoding Transformations

1. **Target Leakage Remediation**: 
   `Performance_Category` is a deterministic categorization of `Productivity_Score`:
   * **Low**: `Productivity_Score` $\le$ 46.64
   * **Medium**: 46.64 < `Productivity_Score` $\le$ 63.03
   * **High**: `Productivity_Score` > 63.03
   
   To avoid target leakage, `Productivity_Score` is dropped from features.

2. **Categorical Variable Conversions**:
   * `Gender` (Nominal) &rarr; One-Hot Encoded (`Gender_Male`, `Gender_Other`), referencing `Female` ($0, 0$).
   * `Part_Time_Job` (Binary) &rarr; `{'No': 0, 'Yes': 1}`.
   * `Internet_Quality` (Ordinal) &rarr; `{'Poor': 0, 'Average': 1, 'Good': 2}`.
   * `Performance_Category` (Ordinal Target) &rarr; `{'Low': 0, 'Medium': 1, 'High': 2}`.
