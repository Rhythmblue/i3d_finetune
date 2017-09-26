import random
import numpy as np
from PIL import Image


class Video_3D:
    def __init__(self,info_list, frame_num, tag='rgb'):
        self.name = info_list[0]
        self.path = info_list[1]
        assert isinstance(info_list[2], int)
        self.total_frame_num = info_list[2]
        assert isinstance(info_list[3],int)

        self.label = info_list[3]

    def get_frames(self, frame_num, side_length=224):
        data = np.zeros(frame_num, side_length, side_length)
        start = random.randint(1, self.total_frame_num)