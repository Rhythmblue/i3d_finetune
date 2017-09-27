from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from rmb_lib.label_trans import *

label_file = '/data4/zhouhao/dataset/ucf101/ucf101_splits/classInd.txt'

def main(img_dir, txt_dir):
    video_list = os.listdir(img_dir)
    video_list.sort()
    label_map = get_label_map(label_file)
    f = open(txt_dir, 'w')
    for video in video_list:
        video_dir = os.path.join(img_dir, video)
        label = trans_label(video.split('_')[1], label_map)
        frames = os.listdir(video_dir)
        length = 0
        for frame in frames:
            if len(frame.split('.')) == 2:
                length += 1
        if len(frame.split('_')) == 2:
            length = int(length/2)
        f.write(video+' '+video_dir+' '+str(length)+' '+str(label)+'\n')
    f.close

if __name__ == '__main__':
    description = 'This script can get the info of videos into a txt.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('img_dir', type=str)
    p.add_argument('txt_dir', type=str)
    main(**vars(p.parse_args()))
