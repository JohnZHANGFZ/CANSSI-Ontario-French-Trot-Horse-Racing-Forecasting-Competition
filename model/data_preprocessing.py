import pandas as pd

df = pd.read_csv('Holdout.csv')

df['IntGender'] = df['Gender'].map({'F': 2, 'M': 1})

df.to_csv('/Users/fangzhengzhang/Desktop/CANSSI/Holdout.csv', index=False)
