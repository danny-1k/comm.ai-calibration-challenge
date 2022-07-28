import torch
from model import CalibModel
from data import data_transforms

import cv2
from PIL import Image
from tqdm import tqdm

import numpy as np

import json

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-data_folder', required=True)
args = parser.parse_args()

data_folder = args.data_folder

net = CalibModel()
net.load_state_dict(torch.load('model.pt'))
net.eval()
net.requires_grad_(False)

transforms = data_transforms['test']

stats = json.load(open('stats.json','r'))

mean = np.array([stats['pitch']['mean'], stats['yaw']['mean']])
std = np.array([stats['pitch']['std'], stats['yaw']['std']])

for i in tqdm(range(5)):

    predictions = []

    f = f'{data_folder}/{i}.hevc'

    print(f)

    cap = cv2.VideoCapture(f)

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = Image.fromarray(frame)

        x = transforms(frame)

        pred = net(x.unsqueeze(0))

        pred = pred.squeeze().numpy()
        pred = (pred * std)+ mean

        predictions.append(pred.squeeze())


    prediction_txt = ''

    for idx,pred in tqdm(enumerate(predictions)):
        prediction_txt+= ' '.join([str(i) for i in pred.tolist()])

        if idx != len(predictions)-1:
            prediction_txt+='\n'

    open(f'test/{i}.txt', 'w').write(prediction_txt)
