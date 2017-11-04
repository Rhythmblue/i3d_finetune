from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import logging
import numpy as np
import tensorflow as tf

import i3d
from rmb_lib.action_dataset import *
from rmb_lib.label_trans import *


_FRAME_SIZE = 224 

_CHECKPOINT_PATHS = {
    'rgb': '/data4/zhouhao/recognition/i3d/model/rgb_910/ucf101_rgb_0.910_model-22260',
    'flow': '/data4/zhouhao/recognition/i3d/model/flow_849/ucf101_flow_0.849_model-24804'
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51
}

log_dir = 'error_record'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow', 'mixed']

    # logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log_'+data_tag+'.txt'), filemode='w', format='%(message)s')

    label_map = get_label_map(os.path.join('data', dataset_name, 'label_map.txt'))

    _, test_info = split_data(
        os.path.join('./data', dataset_name, 'rgb'+'.txt'),
        os.path.join('./data', dataset_name, 'testlist01.txt'))
    _, test_info1 = split_data(
        os.path.join('./data', dataset_name, 'flow'+'.txt'),
        os.path.join('./data', dataset_name, 'testlist01.txt'))

    label_holder = tf.placeholder(tf.int32, [None])
    if data_tag in ['rgb', 'mixed']:
        rgb_data = Action_Dataset(dataset_name, 'rgb', test_info)
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    if data_tag in ['flow', 'mixed']:
        flow_data = Action_Dataset(dataset_name, 'flow', test_info1)
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])


    if data_tag in ['rgb', 'mixed']:
        with tf.variable_scope(_SCOPE['rgb']):
            rgb_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(rgb_holder, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)   
            rgb_fc_out = tf.layers.dense(rgb_logits_dropout, _CLASS_NUM[dataset_name], tf.nn.relu, use_bias=True)
            rgb_top_k_op = tf.nn.in_top_k(rgb_fc_out, label_holder, 1)

    if data_tag in ['flow', 'mixed']:
        with tf.variable_scope(_SCOPE['flow']):
            flow_model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(flow_holder, is_training=False, dropout_keep_prob=1)
            flow_logits_dropout = tf.nn.dropout(flow_logits, 1)   
            flow_fc_out = tf.layers.dense(flow_logits_dropout, _CLASS_NUM[dataset_name], tf.nn.relu, use_bias=True)
            flow_top_k_op = tf.nn.in_top_k(flow_fc_out, label_holder, 1)

    variable_map = {}
    if data_tag in ['rgb', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:
                variable_map[variable.name.replace(':0', '')]=variable
        rgb_saver = tf.train.Saver(var_list=variable_map)
    variable_map = {}
    if data_tag in ['flow', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable_map[variable.name.replace(':0', '')]=variable
        flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    if data_tag == 'rgb':
        fc_out = rgb_fc_out
    if data_tag == 'flow':
        fc_out = flow_fc_out
    if data_tag == 'mixed':
        fc_out = rgb_fc_out + flow_fc_out
    softmax = tf.nn.softmax(fc_out)
    top_k_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if data_tag in ['rgb', 'mixed']:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
    if data_tag in ['flow', 'mixed']:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])

    print('----Here we start!----')
    true_count = 0
    video_size = len(test_info)
    error_record = open(os.path.join(log_dir, 'error_record_'+data_tag+'.txt'), 'w')
    for i in range(video_size):
        feed_dict = {}
        if data_tag in ['rgb', 'mixed']:
            rgb_clip, label = rgb_data.next_batch(
                1, rgb_data.videos[i].total_frame_num, shuffle=False, data_augment=False)
            rgb_clip = rgb_clip/255
            feed_dict[rgb_holder] = rgb_clip
            video_name = rgb_data.videos[i].name
        if data_tag in ['flow', 'mixed']:
            flow_clip, label = flow_data.next_batch(
                1, flow_data.videos[i].total_frame_num, shuffle=False, data_augment=False)
            flow_clip = 2*(flow_clip/255)-1
            feed_dict[flow_holder] = flow_clip
            video_name = flow_data.videos[i].name
        feed_dict[label_holder] = label
        top_1, predictions = sess.run([top_k_op, softmax], feed_dict)
        tmp = np.sum(top_1)
        true_count += tmp
        print('Video%d: %d, accuracy: %.4f (%d/%d) , name:%s' % (i+1, tmp, true_count/video_size, true_count, video_size, video_name))
        # logging.info('Video%d: %d, accuracy: %.4f (%d/%d) , name:%s' % (i+1, tmp, true_count/video_size, true_count, video_size, video_name))
        if tmp==0:
            wrong_answer = np.argmax(predictions, axis=1)[0]
            print('---->answer: %s, probability: %.2f' % (trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
            # logging.info('---->answer: %s, probability: %.2f' % (trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
            error_record.write(
                'video: %s, answer: %s, probability: %.2f\n' % (video_name, trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
    error_record.close()
    accuracy = true_count/ video_size
    print('test accuracy: %.4f' % (accuracy))
    # logging.info('test accuracy: %.4f' % (accuracy))
    sess.close()


if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    main(**vars(p.parse_args()))
