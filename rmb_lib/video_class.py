import random
import os
import numpy as np
from PIL import Image
from data_augment import transform_data


class Video_3D:
    def __init__(self, info_list, tag='rgb', img_format='frame{:06d}{}.jpg'):
        '''
            info_list: [name, path, total_frame, label]
        '''
        self.name = info_list[0]
        self.path = info_list[1]
        if isinstance(info_list[2], int):
            self.total_frame_num = info_list[2]
        else:
            self.total_frame_num = int(info_list[2])
        if isinstance(info_list[3], int):
            self.label = info_list[3]
        else:
            self.label = int(info_list[3])
        self.tag = tag
        self.img_format = img_format

    def get_frames(self, frame_num, side_length=224, is_numpy=True):
        frames = list()
        start = random.randint(1, self.total_frame_num-frame_num+1)
        for i in range(start, start+frame_num):
            frames.extend(self.load_img(i))
        frames = transform_data(frames)
        
        if is_numpy:
            if self.tag=='rgb':
                frames_np = np.zeros(shape=(frame_num, side_length, side_length, 3))
                for i, img in enumerate(frames):
                    frames_np[i] = np.asarray(img)
            elif self.tag=='flow':
                frames_np = np.zeros(shape=(frame_num, side_length, side_length, 2))
                for i in range(0, len(frames), 2):
                    tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
                    frames_np[int(i/2)] = tmp
            return frames_np

        return  frames

    
    def load_img(self, index):
        img_dir = self.path
        if self.tag == 'rgb':
            return [Image.open(os.path.join(img_dir,self.img_format.format(index, ''))).convert('RGB')]
        if self.tag == 'flow':
            u_img = Image.open(os.path.join(img_dir,self.img_format.format(index, '_u'))).convert('L')
            v_img = Image.open(os.path.join(img_dir,self.img_format.format(index, '_v'))).convert('L')
            return [u_img,v_img]
        return

    def __str__(self):
        return '{:s} has {:d} frames, label is {:d}\nPath:{:s}'.format(self.name, self.total_frame_num, self.label, self.path)