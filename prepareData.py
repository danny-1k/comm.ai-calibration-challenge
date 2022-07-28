import json
import pandas as pd


# Standardize 

stats = json.load(open('stats.json','r'))


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


train_df['std_pitch'] = (train_df['pitch']-stats['pitch']['mean'])/ stats['pitch']['std']
train_df['std_yaw'] = (train_df['yaw']-stats['yaw']['mean'])/ stats['yaw']['std']

test_df['std_pitch'] = (test_df['pitch']-stats['pitch']['mean'])/ stats['pitch']['std']
test_df['std_yaw'] = (test_df['yaw']-stats['yaw']['mean'])/ stats['yaw']['std']


needed_columns = ['frame_path','pitch','yaw', 'std_pitch', 'std_yaw']

train_df = train_df[needed_columns]
test_df = test_df[needed_columns]


train_df.to_csv('data/train.csv')
test_df.to_csv('data/test.csv')