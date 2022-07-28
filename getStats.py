import json
import pandas as pd

# stats like mean and std should only be performed on the train data.

train_df = pd.read_csv('data/train.csv')

pitch_mean = train_df['pitch'].mean()
pitch_std = train_df['pitch'].std()

yaw_mean = train_df['yaw'].mean()
yaw_std = train_df['yaw'].std()

stats = {
    'pitch': {
        'mean':pitch_mean,
        'std':pitch_std,
    },
    'yaw' : {
        'mean':yaw_mean,
        'std':yaw_std,
    },
}

json.dump(stats, open('stats.json','w'))