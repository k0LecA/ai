import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import random
from pathlib import Path

data_file = Path("data.csv")

if data_file.exists():
    answer = input("data.csv exists. overwrite? (y/n): ")

    if answer.lower() != 'y':
        exit()

print("Preparing data...")
    


data = []


cats_path='animals/cats'
dogs_path='animals/dogs'

for file in os.listdir(cats_path):
    if file.endswith('.png'):
        data.append(['cat', os.path.join(cats_path, file)])

for file in os.listdir(dogs_path):
    if file.endswith('.png'):
        data.append(['dog', os.path.join(dogs_path, file)])

random.shuffle(data)

total = len(data)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

df = pd.DataFrame(data, columns=['label', 'path'])

df['split'] = ''

df.loc[:train_end - 1, 'split'] = 'train'
df.loc[train_end:val_end - 1, 'split'] = 'validation'
df.loc[val_end:, 'split'] = 'test'
df.to_csv('data.csv', index=False)

print("Data preparation completed and saved to data.csv")
