from __future__ import absolute_import
import argparse
import os
from rmb_lib.label_trans import *


def main(img_dir, txt_dir):
    video_list = os.listdir(img_dir)
    video_list.sort()
    f = open(txt_dir, 'w')
    for video in video_list:
        video_dir = os.path.join(img_dir, video)
        label = trans_label(video.split('_')[1])
        frames = os.listdir(video_dir)
        length = 0
        for frame in frames:
            if len(frame.split('.')) == 2:
                length += 1
        f.write(video+' '+video_dir+' '+str(length)+' '+str(label)+'\n')

if __name__ == '__main__':
    description = 'This script can get the info of videos into a txt.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('img_dir', type=str)
    p.add_argument('txt_dir', type=str)
    main(**vars(p.parse_args()))
