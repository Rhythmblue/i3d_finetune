from rmb_lib.action_dataset import *
import time

info = './data/ucf101/flow.txt'
train_info, _ = split_data(info, './data/ucf101/testlist01.txt')
train = Action_Dataset('ucf101', 'flow', train_info)

start = time.time()
a, b = train.next_batch(20,16)
print(time.time()-start)
print(a.shape)

