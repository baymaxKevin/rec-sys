# coding=utf-8
import tensorflow as tf
from Dice import dice

# item_id, keyword, tag1, tag2, tag3, ks1,ks2
# attention的权重不一样


class Model(object):

    def __init__(self, item_count, keyword_count, tag1_count, tag2_count, tag3_count, ks1_count,
                 ks2_count, user_features_num, item_features_num, user_features_dim, item_features_dim,
                 embedding_size=8, hidden_units=64, deep_layer=[80, 40, 1], lamda=0.1):
        # item_count：item个数
        # keyword_count：关键词个数
        # tag1_count：v4标签1个数
        # tag2_count：v4标签2个数
        # tag3_count：v4标签3个数
        # ks1_count：v4关键词1个数
        # ks2_count：v4关键词2个数
        # user_features_num：用户固有属性个数
        # item_features_num：物品固有属性个数
        # user_features_dim：用户所有固有属性的唯一值个数
        # item_features_dim：物品固有属性的唯一值个数

        # 调参
        # embedding_size ：普通嵌入层神经元个数
        # hidden_units:注意力机制嵌入层的隐藏层神经元个数
        # deep_layer:全连接层的神经元个数

        # 组成
        # user_feature 用户固有属性
        # item_feature 产品固有属性
        # item 当前物品
        # keyword v2关键词
        # tag1 语义标签1
        # tag2 语义标签2
        # tag3 语义标签3
        # ks1 语义关键词1
        # ks2 语义关键词2
        # hist_item 历史记录的物品
        # hist_keyword 历史记录的关键词

        # sl 用户，历史评论产品的个数

        # 输入
        self.y = tf.placeholder(tf.float32, [None, ], name='y')  # [B]
        # [B, user_features_num]
        self.user_feature = tf.placeholder(tf.int32, [None, user_features_num], name='user_feature')
        # [B, item_features_num]
        self.item_feature = tf.placeholder(tf.int32, [None, item_features_num], name='item_feature')

        # self.item = tf.placeholder(tf.int32, [None, ], name='item')  # [B]
        self.item_id = tf.placeholder(tf.int32, [None, ], name='item_id')
        self.keyword = tf.placeholder(tf.int32, [None, ], name='keyword')
        self.tag1 = tf.placeholder(tf.int32, [None, ], name='tag1')
        self.tag2 = tf.placeholder(tf.int32, [None, ], name='tag2')
        self.tag3 = tf.placeholder(tf.int32, [None, ], name='tag3')
        self.ks1 = tf.placeholder(tf.int32, [None, ], name='ks1')
        self.ks2 = tf.placeholder(tf.int32, [None, ], name='ks2')

        # self.hist_item = tf.placeholder(tf.int32, [None, None], name='hist_item')  # [B, T] hist就是history，历史评论过的产品
        self.hist_item = tf.placeholder(tf.int32, [None, None], name='hist_item')  # [B, T]
        self.hist_keyword = tf.placeholder(tf.int32, [None, None], name='hist_keyword')  # [B, T]
        self.hist_tag1 = tf.placeholder(tf.int32, [None, None], name='hist_tag1')  # [B, T]
        self.hist_tag2 = tf.placeholder(tf.int32, [None, None], name='hist_tag2')  # [B, T]
        self.hist_tag3 = tf.placeholder(tf.int32, [None, None], name='hist_tag3')  # [B, T]
        self.hist_ks1 = tf.placeholder(tf.int32, [None, None], name='hist_ks1')  # [B, T]
        self.hist_ks2 = tf.placeholder(tf.int32, [None, None], name='hist_ks2')  # [B, T]

        self.sl = tf.placeholder(tf.int32, [None, ], name='sl')  # [B]
        self.lr = tf.placeholder(tf.float64, [], name='lr')  # [1,1]
        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

        # 变量
        # [user_count，self.embedding_size] user_count一个特征的维度，高维
        user_feature_emb_w = tf.get_variable("user_feature_emb_w",
                                             [user_features_dim,
                                              embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer(
                                                 uniform=False,
                                                 dtype=tf.float32))

        item_feature_emb_w = tf.get_variable("item_feature_emb_w",
                                             [item_features_dim, embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer(
                                                 uniform=False,
                                                 dtype=tf.float32))

        # item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 3])  # 9//2 输出结果 4 , 9.0//2.0 输出结果 4.0
        # 偏差 b
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        item_emb_w = tf.get_variable("item_emb_w",
                                     [item_count, hidden_units],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                      dtype=tf.float32))
        keyword_emb_w = tf.get_variable("keyword_emb_w",
                                        [keyword_count, hidden_units],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                         dtype=tf.float32))
        tag1_emb_w = tf.get_variable("tag1_emb_w",
                                     [tag1_count, hidden_units],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        tag2_emb_w = tf.get_variable("tag2_emb_w",
                                     [tag2_count, hidden_units],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        tag3_emb_w = tf.get_variable("tag3_emb_w",
                                     [tag3_count, hidden_units],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        ks1_emb_w = tf.get_variable("ks1_emb_w",
                                    [ks1_count, hidden_units],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        ks2_emb_w = tf.get_variable("ks2_emb_w",
                                    [ks2_count, hidden_units],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))

        tf.summary.histogram("user_feature_emb_w", user_feature_emb_w)
        tf.summary.histogram("item_feature_emb_w", item_feature_emb_w)
        # tf.summary.histogram("item_emb_w", item_emb_w)
        tf.summary.histogram("item_b", item_b)
        tf.summary.histogram("item_emb_w", item_emb_w)
        tf.summary.histogram("keyword_emb_w", keyword_emb_w)
        tf.summary.histogram("tag1_emb_w", tag1_emb_w)
        tf.summary.histogram("tag2_emb_w", tag2_emb_w)
        tf.summary.histogram("tag3_emb_w", tag3_emb_w)
        tf.summary.histogram("ks1_emb_w", ks1_emb_w)
        tf.summary.histogram("ks2_emb_w", ks2_emb_w)

        # hidden_units = H

        # -- 嵌入层 start ---

        # tf.nn.embedding_lookup(item_emb_w, self.item)  # [B ，hidden_units // 3]
        item_emb = tf.nn.embedding_lookup(item_emb_w, self.item_id)  # [B ，hidden_units // 2] = [B, H // 3]
        keyword_emb = tf.nn.embedding_lookup(keyword_emb_w, self.keyword)
        tag1_emb = tf.nn.embedding_lookup(tag1_emb_w, self.tag1)
        tag2_emb = tf.nn.embedding_lookup(tag2_emb_w, self.tag2)
        tag3_emb = tf.nn.embedding_lookup(tag3_emb_w, self.tag3)
        ks1_emb = tf.nn.embedding_lookup(ks1_emb_w, self.ks1)
        ks2_emb = tf.nn.embedding_lookup(ks2_emb_w, self.ks2)

        i_b = tf.gather(item_b, self.item_id)

        # 在shape【0，1，2】某一个维度上连接
        # tf.nn.embedding_lookup(item_emb_w, self.hist_item) # [B, T, hidden_units // 3]
        hist_item_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_item)  # [B, T, hidden_units // 3]
        hist_keyword_emb = tf.nn.embedding_lookup(keyword_emb_w, self.hist_keyword)  # [B, T, hidden_units // 3]
        hist_tag1_emb = tf.nn.embedding_lookup(tag1_emb_w, self.hist_tag1)  # [B, T, hidden_units // 3]
        hist_tag2_emb = tf.nn.embedding_lookup(tag2_emb_w, self.hist_tag2)  # [B, T, hidden_units // 3]
        hist_tag3_emb = tf.nn.embedding_lookup(tag3_emb_w, self.hist_tag3)  # [B, T, hidden_units // 3]
        hist_ks1_emb = tf.nn.embedding_lookup(ks1_emb_w, self.hist_ks1)  # [B, T, hidden_units // 3]
        hist_ks2_emb = tf.nn.embedding_lookup(ks2_emb_w, self.hist_ks2)  # [B, T, hidden_units // 3]
        # [B, T, H]
        # -- 嵌入层 end ---

        # -- attention start ---
        hist_item = attention(item_emb, hist_item_emb, self.sl)  # [B, 1, H]
        hist_item = tf.layers.batch_normalization(inputs=hist_item)
        hist_item = tf.reshape(hist_item, [-1, hidden_units])  # [B, H]
        hist_item = tf.layers.dense(hist_item, hidden_units)  # [B, H]

        hist_keyword = attention(keyword_emb, hist_keyword_emb, self.sl)  # [B, 1, H]
        hist_keyword = tf.layers.batch_normalization(inputs=hist_keyword)
        hist_keyword = tf.reshape(hist_keyword, [-1, hidden_units])  # [B, H]
        hist_keyword = tf.layers.dense(hist_keyword, hidden_units)  # [B, H]

        hist_tag1 = attention(tag1_emb, hist_tag1_emb, self.sl)  # [B, 1, H]
        hist_tag1 = tf.layers.batch_normalization(inputs=hist_tag1)
        hist_tag1 = tf.reshape(hist_tag1, [-1, hidden_units])  # [B, H]
        hist_tag1 = tf.layers.dense(hist_tag1, hidden_units)  # [B, H]

        hist_tag2 = attention(tag2_emb, hist_tag2_emb, self.sl)  # [B, 1, H]
        hist_tag2 = tf.layers.batch_normalization(inputs=hist_tag2)
        hist_tag2 = tf.reshape(hist_tag2, [-1, hidden_units])  # [B, H]
        hist_tag2 = tf.layers.dense(hist_tag2, hidden_units)  # [B, H]

        hist_tag3 = attention(tag3_emb, hist_tag3_emb, self.sl)  # [B, 1, H]
        hist_tag3 = tf.layers.batch_normalization(inputs=hist_tag3)
        hist_tag3 = tf.reshape(hist_tag3, [-1, hidden_units])  # [B, H]
        hist_tag3 = tf.layers.dense(hist_tag3, hidden_units)  # [B, H]

        hist_ks1 = attention(ks1_emb, hist_ks1_emb, self.sl)  # [B, 1, H]
        hist_ks1 = tf.layers.batch_normalization(inputs=hist_ks1)
        hist_ks1 = tf.reshape(hist_ks1, [-1, hidden_units])  # [B, H]
        hist_ks1 = tf.layers.dense(hist_ks1, hidden_units)  # [B, H]

        hist_ks2 = attention(ks2_emb, hist_ks2_emb, self.sl)  # [B, 1, H]
        hist_ks2 = tf.layers.batch_normalization(inputs=hist_ks2)
        hist_ks2 = tf.reshape(hist_ks2, [-1, hidden_units])  # [B, H]
        hist_ks2 = tf.layers.dense(hist_ks2, hidden_units)  # [B, H]

        # -- attention end ---

        # -- 普通嵌入层 --
        user_feature = tf.nn.embedding_lookup(user_feature_emb_w,
                                              self.user_feature)  # [B, embedding_size, user_features_num]
        user_feature = tf.reshape(user_feature, [-1, embedding_size * user_features_num])

        item_feature = tf.nn.embedding_lookup(item_feature_emb_w,
                                              self.item_feature)  # [B, embedding_size, user_features_num]
        item_feature = tf.reshape(item_feature, [-1, embedding_size * item_features_num])

        # -- 普通嵌入层 --

        # -- fcn begin -------
        # -- 训练集全连接层 开始 -------
        din_i = tf.concat(
            [hist_item, hist_keyword, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2,
             item_emb, keyword_emb, tag1_emb, tag2_emb, tag3_emb, ks1_emb, ks2_emb,
             user_feature, item_feature], axis=-1)

        d_layer_1_i = tf.layers.dense(din_i, deep_layer[0], activation=None, name='f1')  # 全连接层  [B, 80]
        d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
        d_layer_1_i = tf.nn.dropout(d_layer_1_i, self.keep_prob)
        d_layer_1_i = tf.layers.batch_normalization(inputs=d_layer_1_i, name='b1')

        d_layer_2_i = tf.layers.dense(d_layer_1_i, deep_layer[1], activation=None, name='f2')
        d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
        d_layer_2_i = tf.nn.dropout(d_layer_2_i, self.keep_prob)
        d_layer_2_i = tf.layers.batch_normalization(inputs=d_layer_2_i, name='b2')

        d_layer_3_i = tf.layers.dense(d_layer_2_i, deep_layer[2], activation=None, name='f3')

        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])  # 展开成行向量

        self.logits = i_b + d_layer_3_i
        self.sig_logits = tf.sigmoid(self.logits, name='sig_logits')
        # -- 训练集全连接层 结束 -------

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        with tf.name_scope('loss'):  # 损失
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y),
                                       name='loss') + tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr, name='adam')
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess, merge, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2,
    hist_item, hist_keyword, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl),
              l, keep_prob):
        summary, loss, _ = sess.run([merge, self.loss, self.train_op], feed_dict={
            self.y: y,

            self.user_feature: user_feature,
            self.item_feature: item_feature,
            self.item_id: item_id,
            self.keyword: keyword,
            self.tag1: tag1,
            self.tag2: tag2,
            self.tag3: tag3,
            self.ks1: ks1,
            self.ks2: ks2,

            self.hist_item: hist_item,
            self.hist_keyword: hist_keyword,
            self.hist_tag1: hist_tag1,
            self.hist_tag2: hist_tag2,
            self.hist_tag3: hist_tag3,
            self.hist_ks1: hist_ks1,
            self.hist_ks2: hist_ks2,

            self.sl: sl,
            self.lr: l,
            self.keep_prob: keep_prob
        })

        return summary, loss

    def eval_train(self, sess, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2,
    hist_item, hist_keyword, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl,
    keep_prob)):
        sig_logits, loss = sess.run([self.sig_logits, self.loss], feed_dict={
            self.y: y,

            self.user_feature: user_feature,
            self.item_feature: item_feature,
            self.item_id: item_id,
            self.keyword: keyword,
            self.tag1: tag1,
            self.tag2: tag2,
            self.tag3: tag3,
            self.ks1: ks1,
            self.ks2: ks2,

            self.hist_item: hist_item,
            self.hist_keyword: hist_keyword,
            self.hist_tag1: hist_tag1,
            self.hist_tag2: hist_tag2,
            self.hist_tag3: hist_tag3,
            self.hist_ks1: hist_ks1,
            self.hist_ks2: hist_ks2,

            self.sl: sl,
            self.keep_prob: keep_prob
        })
        return sig_logits, loss

    def eval(self, sess, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2, hist_item,
    hist_keyword, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl), keep_prob):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.y: y,

            self.user_feature: user_feature,
            self.item_feature: item_feature,
            self.item_id: item_id,
            self.keyword: keyword,
            self.tag1: tag1,
            self.tag2: tag2,
            self.tag3: tag3,
            self.ks1: ks1,
            self.ks2: ks2,

            self.hist_item: hist_item,
            self.hist_keyword: hist_keyword,
            self.hist_tag1: hist_tag1,
            self.hist_tag2: hist_tag2,
            self.hist_tag3: hist_tag3,
            self.hist_ks1: hist_ks1,
            self.hist_ks2: hist_ks2,

            self.sl: sl,
            self.keep_prob: keep_prob
        })
        return u_auc, socre_p_and_n

    def test(self, sess, merge, (y, user_feature, item_feature, item_id, keyword, tag1, tag2, tag3, ks1, ks2,
    hist_item, hist_keyword, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl,
    keep_prob)):
        summary, sig_logits, loss = sess.run([merge, self.sig_logits, self.loss], feed_dict={
            self.y: y,

            self.user_feature: user_feature,
            self.item_feature: item_feature,
            self.item_id: item_id,
            self.keyword: keyword,
            self.tag1: tag1,
            self.tag2: tag2,
            self.tag3: tag3,
            self.ks1: ks1,
            self.ks2: ks2,

            self.hist_item: hist_item,
            self.hist_keyword: hist_keyword,
            self.hist_tag1: hist_tag1,
            self.hist_tag2: hist_tag2,
            self.hist_tag3: hist_tag3,
            self.hist_ks1: hist_ks1,
            self.hist_ks2: hist_ks2,

            self.sl: sl,
            self.keep_prob: keep_prob
        })
        return summary, sig_logits, loss

    def save(self, sess, path, saver):
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def attention(queries, keys, keys_length):
    '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]  # queries_hidden_units = H
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [B, H * T]
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # [B, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # [B, T, 4H]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.relu, name='f1_att', reuse=tf.AUTO_REUSE)  # [B, T, 80]
    # d_layer_1_all = dice(d_layer_1_all, name='dice_1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.relu, name='f2_att', reuse=tf.AUTO_REUSE)  # [B, T, 40]
    # d_layer_2_all = dice(d_layer_2_all, name='dice_2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)  # [B, T, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # [B, 1, T]
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs
