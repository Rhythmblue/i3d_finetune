import numpy as np
from rmb_lib.video_3d import Video_3D


class Action_Dataset:
    def __init__(self, name, tag, video_info):
        self.dataset_name = name
        self.tag = tag
        self.videos = [Video_3D(x, self.tag) for x in video_info]
        self.size = len(self.videos)
        self.epoch_completed = 0
        self.index_in_epoch = 0
        self.perm = np.arange(self.size)


    def next_batch(self, batch_size, frame_num, shuffle=True, data_augment=True):
        start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch = end % self.size
        batch = []
        label = []
        if end >= self.size:
            self.epoch_completed += 1
            for i in range(start, self.size):
                batch.append(
                    self.videos[self.perm[i]].get_frames(frame_num, data_augment=data_augment))
                label.append(self.videos[self.perm[i]].label)
            if shuffle:
                np.random.shuffle(self.perm)
            for i in range(0, self.index_in_epoch):
                batch.append(
                    self.videos[self.perm[i]].get_frames(frame_num, data_augment=data_augment))
                label.append(self.videos[self.perm[i]].label)
        else:
            for i in range(start, end):
                batch.append(
                    self.videos[self.perm[i]].get_frames(frame_num, data_augment=data_augment))
                label.append(self.videos[self.perm[i]].label)
        return np.stack(batch), np.stack(label)


def split_data(data_info, test_split):
    f1 = open(data_info)
    f2 = open(test_split)
    test = list()
    train_info = list()
    test_info = list()
    for line in f2.readlines():
        test.append(line.split('/')[1].split('.')[0])
    for line in f1.readlines(): 
        info = line.strip().split(' ')
        if info[0] in test:
            test_info.append(info)
        else:
            train_info.append(info)
    f1.close()
    f2.close()
    return train_info, test_info
