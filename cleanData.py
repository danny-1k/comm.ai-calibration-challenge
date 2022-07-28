import pandas as pd

df = pd.read_csv('data/data.csv')
#Drop rows with NaNs in any of the columns

df.dropna(axis=0,inplace=True)

df.to_csv('data/data.csv')
