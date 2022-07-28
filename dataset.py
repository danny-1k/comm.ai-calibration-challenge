import cv2
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

class DatasetGenerator:
    def __init__(self, datadir, savedir):

        self.datadir = datadir
        self.savedir = savedir
        self.frame_paths = []


        if not os.path.exists(datadir):
            raise ValueError(f'Datadir does not exist. Got {datadir}')

        if not os.path.exists(savedir):
            print('savedir does not exist... creating.')
            os.makedirs(os.path.join(savedir, 'videos'))

        elif not os.path.exists(os.path.join(savedir, 'videos')):
            print('Videos directory does not exist... creating.')
            os.makedirs(os.path.join(savedir, 'videos'))

        self.df = pd.DataFrame(columns=['frame_path', 'pitch', 'yaw'])

    def generate(self,):
        print('Started ...')

        videos = sorted([f for f in os.listdir(self.datadir) if f.endswith('hevc')])
        all_labels = sorted([f for f in os.listdir(self.datadir) if f.endswith('txt')])

        for idx,(video, labels) in enumerate(zip(videos, all_labels)):
            video = os.path.join(self.datadir, video)
            labels = open(os.path.join(self.datadir,labels),'r').read()
            self.generatedata(video, idx,labels)


        self.save_data_frame()



    def generatedata(self, video, video_id, labels):
        labels = labels.split('\n')
        no_labels = len(labels)

        cap = cv2.VideoCapture(video)

        for n in tqdm(range(no_labels)):
            try:
                label = labels[n]

                pitch = float(label.split(' ')[0])
                yaw = float(label.split(' ')[1])

                frame = self.extract_frame(cap)
                frame_path = self.save_frame(frame, video_id, n)

                new_row = pd.DataFrame(columns=['frame_path', 'pitch', 'yaw'])
                new_row['frame_path'] = pd.Series([frame_path])
                new_row['pitch'] = pd.Series([pitch])
                new_row['yaw'] = pd.Series([yaw])

                self.df = self.df.append(new_row,ignore_index=True)
            except:
                continue



    def extract_frame(self, cap):
        _, frame = cap.read()
        return frame

    def save_frame(self,frame, video_id, frame_id):
        folder_path = os.path.join(self.savedir,'videos',f'{video_id}')


        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        img_path = os.path.join(folder_path, f'frame_{frame_id}.png')

        if os.path.exists(img_path):
            return img_path
        
        img = self.to_pil_img(frame)
        img.save(img_path)
        
        return img_path

    def to_pil_img(self,arr):
        img = Image.fromarray(arr)
        return img


    def save_data_frame(self):
        self.df.to_csv(os.path.join(self.savedir,'data.csv'))