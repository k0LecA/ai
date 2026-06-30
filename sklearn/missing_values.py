import pandas as pd
import numpy as np

#df = pd.read_csv('Student_Productivity_Dataset.csv')
#df = pd.read_csv('filled_data.csv')
df = pd.read_csv('normalized_data.csv')

print(df.isnull().sum())
