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


_BATCH_SIZE = 10
_CLIP_SIZE = 16
_FRAME_SIZE = 224 
_LEARNING_RATE = 0.001
_GLOBAL_EPOCH = 40


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
tf.train.MomentumOptimizer
log_dir = './model'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow']
    train_info, test_info = split_data(
        os.path.join('./data', dataset_name, data_tag+'.txt'),
        os.path.join('./data', dataset_name, 'testlist01.txt'))
    train_data = Action_Dataset(dataset_name, data_tag, train_info)
    test_data = Action_Dataset(dataset_name, data_tag, test_info)
    with open(os.path.join('./data', dataset_name, 'label_map.txt')) as f:
        label_map = [x.strip() for x in f.readlines()]

    clip_holder = tf.placeholder(
        tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL[train_data.tag]])
    label_holder = tf.placeholder(tf.int32, [None])
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)
    #tf.summary.image("demo", clip_holder[0], 6)
  
    with tf.variable_scope(_SCOPE[train_data.tag]):
        model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
        logits, _ = model(clip_holder, is_training=is_train_holder, dropout_keep_prob=dropout_holder)
        logits_dropout = tf.nn.dropout(logits, dropout_holder)   
        fc_out = tf.layers.dense(logits_dropout, 101, tf.nn.relu, use_bias=True)
        top_k_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    variable_map = {}
    train_var = []
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        #if tmp[1] == 'dense':
        #    train_var.append(variable)
        if tmp[0] == _SCOPE[train_data.tag] and tmp[1] != 'dense':
            variable_map[variable.name.replace(':0', '')] = variable
        if tmp[-1] == 'w:0' or tmp[-1] == 'kernel:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    saver2 = tf.train.Saver(max_to_keep=10)

    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=fc_out))
    total_loss = loss + 5e-7 * loss_weight
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_weight', loss_weight)
    tf.summary.scalar('total_loss', total_loss)

    per_epoch_step = int(np.ceil(train_data.size/_BATCH_SIZE))
    global_step = _GLOBAL_EPOCH * per_epoch_step
    decay_step = int(2*per_epoch_step)
    global_index = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        _LEARNING_RATE, global_index, decay_step, 0.7, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, global_step=global_index)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, _CHECKPOINT_PATHS[train_data.tag+'_imagenet'])
    print('----Here we start!----')

    step = 0
    true_count = 0
    tmp_count = 0
    accuracy_tmp = 0
    per_test_step = int(np.floor(test_data.size/_BATCH_SIZE))
    while step <= global_step:
        step += 1
        start_time = time.time()
        clip, label = train_data.next_batch(_BATCH_SIZE, _CLIP_SIZE)
        if train_data.tag == 'rgb':
            clip = clip/255
        else:
            clip = 2*(clip/255)-1
        _, loss_now, loss_plus, predictions, summary = sess.run([optimizer, total_loss, loss_weight, top_k_op, merged],
                               feed_dict={clip_holder: clip,
                                          label_holder: label,
                                          dropout_holder: 0.7,
                                          is_train_holder: True})
        duration = time.time() - start_time
        tmp = np.sum(predictions)
        true_count += tmp
        tmp_count += tmp
        train_writer.add_summary(summary, step)
        if step % 20 == 0:
            accuracy = tmp_count / (20*_BATCH_SIZE)
            print('step: %-4d, loss: %-.4f, accuracy: %.3f (%.2f sec/batch)' % (step, loss_now, accuracy, float(duration)))
            tmp_count = 0
            # print(label)
            # print(predictions)
        if step % per_epoch_step ==0:
            accuracy = true_count/ (per_epoch_step*_BATCH_SIZE)
            print('Epoch%d, train accuracy: %.3f' % (train_data.epoch_completed, accuracy))
            true_count = 0
        if step % (2*per_epoch_step) == 0:
            true_count = 0
            for i in range(per_test_step):
                fc_tmp = np.zeros([_BATCH_SIZE,101])
                for j in range(3):
                    index = test_data.index_in_epoch
                    clip, label = test_data.next_batch(
                        _BATCH_SIZE, _CLIP_SIZE, shuffle=False, data_augment=False)
                    if(j != 2):
                        test_data.index_in_epoch = index
                    if train_data.tag == 'rgb':
                        clip = clip/255
                    else:
                        clip = 2*(clip/255)-1
                    fc_now = sess.run(fc_out, feed_dict={clip_holder: clip,
                                                        label_holder: label,
                                                        dropout_holder: 1,
                                                        is_train_holder: False})
                    fc_tmp += fc_now
                predictions = np.argmax(fc_tmp, 1)
                true_count += np.sum(predictions == label)
            accuracy = true_count/ (per_test_step*_BATCH_SIZE)
            true_count = 0
            test_data.index_in_epoch = 0
            print('Epoch%d, test accuracy: %.3f' % (train_data.epoch_completed, accuracy))
            if accuracy > 0.80:
                if accuracy>accuracy_tmp:
                    accuracy_tmp = accuracy
                    saver2.save(sess, log_dir + '/ucf101_'+train_data.tag+'_{:.3f}_model'.format(accuracy), step)
    train_writer.close()
    sess.close()


if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    main(**vars(p.parse_args()))
