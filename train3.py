import os
import gc
import time
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from input import DataInput
from model import Model
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import math
prefix = os.path.abspath(os.path.dirname(os.getcwd()))
random.seed(1234)


def split_to_int(x):
    if x == x:
        arr = [int(i) for i in x.split(",")]
        return arr
    else:
        return ''


data = pd.read_csv(prefix + '/data/train_set.csv', sep=',')
print('total_count', data.shape[0])
del data['batchNo']
train_set = data[data['userId'] % 50 == 7]
test_set = data[data['userId'] % 50 == 8]
del train_set['userId']
del test_set['userId']
del data
test_set = test_set.sample(frac=0.2)
gc.collect()
train_set['user_feature'] = train_set['user_feature'].apply(lambda x: [int(i) for i in x.split(",")])
train_set['item_feature'] = train_set['item_feature'].apply(lambda x: [int(i) for i in x.split(",")])
train_set['userItemHistory'] = train_set['userItemHistory'].apply(lambda x: split_to_int(x))
train_set['userKeywordHistory'] = train_set['userKeywordHistory'].apply(lambda x: split_to_int(x))
train_set['userTag1History'] = train_set['userTag1History'].apply(lambda x: split_to_int(x))
train_set['userTag2History'] = train_set['userTag2History'].apply(lambda x: split_to_int(x))
train_set['userTag3History'] = train_set['userTag3History'].apply(lambda x: split_to_int(x))
train_set['userKs1History'] = train_set['userKs1History'].apply(lambda x: split_to_int(x))
train_set['userKs2History'] = train_set['userKs2History'].apply(lambda x: split_to_int(x))
test_set['user_feature'] = test_set['user_feature'].apply(lambda x: [int(i) for i in x.split(",")])
test_set['item_feature'] = test_set['item_feature'].apply(lambda x: [int(i) for i in x.split(",")])
test_set['userItemHistory'] = test_set['userItemHistory'].apply(lambda x: split_to_int(x))
test_set['userKeywordHistory'] = test_set['userKeywordHistory'].apply(lambda x: split_to_int(x))
test_set['userTag1History'] = test_set['userTag1History'].apply(lambda x: split_to_int(x))
test_set['userTag2History'] = test_set['userTag2History'].apply(lambda x: split_to_int(x))
test_set['userTag3History'] = test_set['userTag3History'].apply(lambda x: split_to_int(x))
test_set['userKs1History'] = test_set['userKs1History'].apply(lambda x: split_to_int(x))
test_set['userKs2History'] = test_set['userKs2History'].apply(lambda x: split_to_int(x))

train_set = train_set.values
test_set = test_set.values
print('train_set_count', train_set.shape)
print('test_set_count', test_set.shape)

train_batch_size = 128
eval_batch_size = 5000
slice_size = 5000
batch_step = math.ceil(train_set.shape[0] / (train_batch_size*1.0))


def read_dict(path):
    with open(prefix + path) as json_file:
        return json.load(json_file)


def _eval(sess, merge, model, total_step, test_writer):
    score_arr = []
    loss_sum = 0
    slice_test_set = np.array(random.sample(test_set, slice_size))
    for i, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2, hist_item, hist_keyword,
            hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl) in DataInput(
            slice_test_set, eval_batch_size):
        summary, score, loss = model.test(sess, merge,
                                          (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3,
                                           ks1, ks2, hist_item, hist_keyword, hist_tag1, hist_tag2,
                                           hist_tag3, hist_ks1, hist_ks2, sl, 1.0))
        test_writer.add_summary(summary, global_step=total_step)
        loss_sum = loss_sum + loss
        score_arr += list(score)
    true_y = slice_test_set[:, 0]
    true_y = list(true_y)

    score_arr_binary = [int(i > 0.5) for i in score_arr]
    auc = roc_auc_score(true_y, score_arr)
    recall = recall_score(true_y, score_arr_binary)
    precision = precision_score(true_y, score_arr_binary)
    accuracy = accuracy_score(true_y, score_arr_binary)

    # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
    #                                                                 output_node_names=['sig_logits'])
    # with tf.gfile.FastGFile('save_path/din.pb' + '.' + str(epoch) + '.' + str(step), mode='wb') as f:
    #     f.write(output_graph_def.SerializeToString())

    return auc, recall, precision, accuracy, (loss_sum * eval_batch_size) / slice_size


def _eval_train(sess, model):
    score_arr = []
    loss_sum = 0
    slice_train_set = np.array(random.sample(train_set, slice_size))
    for _, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2, hist_item, hist_keyword,
            hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl) in DataInput(
            slice_train_set, eval_batch_size):
        score, loss = model.eval_train(sess, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3,
                                              ks1, ks2, hist_item, hist_keyword, hist_tag1, hist_tag2,
                                              hist_tag3, hist_ks1, hist_ks2, sl, 1.0))
        loss_sum = loss_sum + loss
        score_arr += list(score)
    true_y = slice_train_set[:, 0]
    true_y = list(true_y)

    score_arr_binary = [int(i > 0.5) for i in score_arr]
    auc = roc_auc_score(true_y, score_arr)
    recall = recall_score(true_y, score_arr_binary)
    precision = precision_score(true_y, score_arr_binary)
    accuracy = accuracy_score(true_y, score_arr_binary)

    return auc, recall, precision, accuracy, (loss_sum * eval_batch_size) / slice_size


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(device_count={"CPU": 32},
                        inter_op_parallelism_threads=0,
                        intra_op_parallelism_threads=0,
                        gpu_options=gpu_options)


def run():
    with tf.Session(config=config) as sess:
        count_dict = read_dict('/data/count_dict.txt')
        count_dict['lamda'] = 0.1
        model = Model(**count_dict)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        merge = tf.summary.merge_all()
        if tf.gfile.Exists("logs"):
            tf.gfile.DeleteRecursively("logs")
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs/test")
        # model export
        saver = tf.train.Saver()
        inputs = {
            "user_feature": tf.saved_model.utils.build_tensor_info(model.user_feature),
            "hist_item": tf.saved_model.utils.build_tensor_info(model.hist_item),
            "hist_keyword": tf.saved_model.utils.build_tensor_info(model.hist_keyword),
            "hist_tag1": tf.saved_model.utils.build_tensor_info(model.hist_tag1),
            "hist_tag2": tf.saved_model.utils.build_tensor_info(model.hist_tag2),
            "hist_tag3": tf.saved_model.utils.build_tensor_info(model.hist_tag3),
            "hist_ks1": tf.saved_model.utils.build_tensor_info(model.hist_ks1),
            "hist_ks2": tf.saved_model.utils.build_tensor_info(model.hist_ks2),
            "sl": tf.saved_model.utils.build_tensor_info(model.sl),
            "item_feature": tf.saved_model.utils.build_tensor_info(model.item_feature),
            "item_id": tf.saved_model.utils.build_tensor_info(model.item_id),
            "keyword": tf.saved_model.utils.build_tensor_info(model.keyword),
            "tag1": tf.saved_model.utils.build_tensor_info(model.tag1),
            "tag2": tf.saved_model.utils.build_tensor_info(model.tag2),
            "tag3": tf.saved_model.utils.build_tensor_info(model.tag3),
            "ks1": tf.saved_model.utils.build_tensor_info(model.ks1),
            "ks2": tf.saved_model.utils.build_tensor_info(model.ks2),
            "keep_prob": tf.saved_model.utils.build_tensor_info(model.keep_prob)
        }
        outputs = {
            "sig_logits": tf.saved_model.utils.build_tensor_info(model.sig_logits)
        }
        signature_def_map = {
            "predict": tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )}
        # model_exporter = tf.contrib.session_bundle.exporter.Exporter(saver)
        # model_exporter.init(sess.graph.as_graph_def(),
        #                     named_graph_signatures={
        #         'inputs': tf.contrib.session_bundle.exporter.generic_signature(inputs),
        #         'outputs': tf.contrib.session_bundle.exporter.generic_signature(outputs)})

        sys.stdout.flush()
        lr = 0.01
        keep_prob = 1.0
        start_time = time.time()
        for epoch in range(10):
            # random.shuffle(train_set)
            loss_sum = 0.0
            for i, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2, hist_item,
                    hist_keyword, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl) in \
                    DataInput(train_set, train_batch_size):
                step = epoch * batch_step + i
                # if step / 2000 % 3 == 0:
                #     lr = 0.001
                # if step / 2000 % 3 == 1:
                #     lr = 0.0005
                # if step / 2000 % 3 == 2:
                #     lr = 0.0001
                train_summary, loss = model.train(sess, merge,
                                                  (y, user_feature, item_feature, item_id, keyword, tag1, tag2,
                                                   tag3, ks1, ks2, hist_item, hist_keyword, hist_tag1,
                                                   hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl), lr, keep_prob)

                loss_sum += loss
                if i % 100 == 0:

                    test_auc, test_recall, test_precision, test_accuracy, test_loss = _eval(sess, merge, model,
                                                                                            step, test_writer)
                    train_auc, train_recall, train_precision, train_accuracy, train_loss = _eval_train(sess, model)
                    print('epoch %d step %d,train:b_loss:%.4f loss:%.4f auc:%.4f recall:%.4f pre:%.4f acc:%.4f' %
                          (epoch, i, loss_sum / 100, train_loss, train_auc, train_recall, train_precision,
                           train_accuracy))
                    print('epoch %d step %d,test:               loss:%.4f auc:%.4f recall:%.4f pre:%.4f acc:%.4f' %
                          (epoch, i, test_loss, test_auc, test_recall, test_precision, test_accuracy))
                    sys.stdout.flush()
                    loss_sum = 0.0
                    saver.save(sess, 'save_path/ckpt')
                    builder = tf.saved_model.builder.SavedModelBuilder('model_path/' + str(int(step)))
                    builder.add_meta_graph_and_variables(sess,
                                                         [tf.saved_model.tag_constants.SERVING],
                                                         signature_def_map=signature_def_map)
                    builder.save()
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                    output_node_names=['sig_logits'])
                    with tf.gfile.FastGFile('save_path/din.pb' + '.' + str(epoch) + '.' + str(i), mode='wb') as f:
                        f.write(output_graph_def.SerializeToString())
                else:
                    train_writer.add_summary(train_summary, global_step=step)

            print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time() - start_time))
            sys.stdout.flush()
        sys.stdout.flush()


run()
