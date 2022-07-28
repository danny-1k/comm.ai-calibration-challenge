import random
import pandas as pd

random.seed(44)

df =  pd.read_csv('data/data.csv')
no_records = len(df)

indices = list(range(no_records))
random.shuffle(indices)

train_ratio = .8
no_train = int(train_ratio*no_records)

train_indices =  indices[:no_train] # first `no_train` samples
test_indices = indices[no_train:]

train_df = df.iloc[train_indices]
test_df = df.iloc[test_indices]

train_df.to_csv('data/train.csv')
test_df.to_csv('data/test.csv')
