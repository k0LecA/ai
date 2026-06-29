import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('filled_data.csv')

sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

#later mb