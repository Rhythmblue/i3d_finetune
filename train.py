from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import numpy as np
import tensorflow as tf

import i3d 
from rmb_lib.action_dataset import *


_BATCH_SIZE = 16
_CLIP_SIZE = 32
_FRAME_SIZE = 224 

_CHECKPOINT_PATHS = {
    'rgb': './data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './data/checkpoints/flow_imagenet/model.ckpt',
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}


def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow']
    train_info, test_info = split_data(
        os.path.join('./data', dataset_name, data_tag+'.txt'),
        os.path.join('./data', dataset_name, 'testlist01.txt'))
    train_data = Action_Dataset(dataset_name, data_tag, train_info)
    #test_data = Action_Dataset(dataset_name, data_tag, test_info)
    with open(os.path.join('./data', dataset_name, 'label_map.txt')) as f:
        label_map = [x.strip() for x in f.readlines()]

    clip_holder = tf.placeholder(
        tf.float32, [None, _CLIP_SIZE, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL[train_data.tag]])
    label_holder = tf.placeholder(tf.int32, [None])
  
    with tf.variable_scope(_SCOPE[train_data.tag]):
        model = i3d.InceptionI3d(len(label_map), spatial_squeeze=True, final_endpoint='Logits')
        logits, _ = model(clip_holder, is_training=True, dropout_keep_prob=0.5)    

    variable_map = {}
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[0] == _SCOPE[train_data.tag] and tmp[2] != 'Logits':
            variable_map[variable.name.replace(':0', '')] = variable
        if tmp[-1] == 'w:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    prediction = tf.nn.softmax(logits)

    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label_holder))
    total_loss = loss + 1e-7 * loss_weight
    tf.summary.scalar('total_loss', total_loss)

    learning_rate = tf.train.exponential_decay(0.1, 3e3, 1e3, 0.1, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    step = 0
    while step <= 3e3:
        step += 1
        start_time = time.time()
        clip, label = train_data.next_batch(_BATCH_SIZE, _CLIP_SIZE)
        _, loss_now = sess.run([optimizer, total_loss],
                               feed_dict={clip_holder: clip,
                                          label_holder: label})
        duration = time.time() - start_time

        print('step: %-4d, loss: %-.4f (%.2f sec/batch)' % (step, loss_now, float(duration)))



if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type = str)
    main(**vars(p.parse_args()))
