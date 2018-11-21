# coding=utf-8

import tensorflow as tf

'''
本模块为tensorflow工具包
    function
        add_embedding                             -- 添加embedding
        data_mean                                 -- 算句子的向量表示，用词算平均值的方式
        create_learning_rate                      -- 指数下降学习率
        check_accuracy                            -- 计算正确率
        train_optimizer                           -- 训练
        add_loss_op                               -- 添加损失函数操作
'''

def add_embedding(embedding_matrix, x_ph_ids):
    '''
    添加embedding
    :param embedding_matrix: embedding矩阵
    :param x_ph_ids: x_placeholder
    :return: embedding变量和用embedding表示过得向量
    '''
    embedding = tf.Variable(embedding_matrix, name='Embedding', trainable=True)
    x_ph_embedding = tf.nn.embedding_lookup(embedding, x_ph_ids)
    return embedding, x_ph_embedding


def data_mean(x_batch, sent_len_batch, dim):
    '''
    算句子的向量表示，用词算平均值的方式
    :param x_batch:一批句子表示
    :param sent_len_batch: 句子长度
    :param dim: 轴
    :return: 一批句子向量
    '''
    # 句子的限制长度
    limit_len = tf.shape(x_batch)[dim]
    # -------mask----------
    mask = tf.sequence_mask(sent_len_batch, limit_len)
    mask_reshape = tf.reshape(mask, shape=(-1, limit_len, 1))
    x_batch_mask = x_batch * tf.cast(mask_reshape, dtype=x_batch.dtype)
    # ------------------------
    sum = tf.reduce_sum(x_batch_mask, dim)
    sent_len_batch_rank = tf.reshape(sent_len_batch, shape=(-1, 1))
    avg = sum / (tf.to_float(sent_len_batch_rank) + 1e-20)
    return avg

def create_learning_rate(learning_rate_start,global_step, decay_steps, decay_rate , staircase=True):
    '''
        指数下降学习率
        learning_rate = start_rate * (decay_rate)^(global_step/decay_steps)

        公式中，learning_rate： 当前的学习速率
        start_rate：最初的学习速率
        decay_rate：每轮学习的衰减率，0<decay_rate<1
        global_step：当前的学习步数，等同于我们将 batch 放入学习器的次数
        decay_steps：每轮学习的步数，decay_steps = sample_size/batch  即样本总数除以每个batch的大小
        staircase： 是否采用阶梯形式
    '''
    learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
    return learning_rate

def check_accuracy(y_true, y_pre):
    '''
    正确率
    :param y_true:真是值
    :param y_pre:预测值
    :return:正确率
    '''
    count_equal = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_true, 1))
    count_equal_float = tf.cast(count_equal, tf.float32)
    accuracy = tf.reduce_mean(count_equal_float)
    return accuracy

def train_optimizer(loss, learning_rate_start = 0.01,decay_steps = None,decay_rate = None, algorithm = 'SGD'):
    '''
    训练
    :param loss: 损失函数值
    :param learning_rate_start: 初始学习率
    :param decay_steps:每轮学习的步数
    :param decay_rate:每轮学习的衰减率
    :param algorithm: 算法可选 SGD Adam Adgrad
    :return:
            global_step:训练的总步数
            train_step：Optimizer
            learning_rate：当前学习率
    '''

    # global_step用于记录总训练次数
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = create_learning_rate(learning_rate_start, global_step, decay_steps, decay_rate)
    optimizer = None
    # 优化方法选择
    if algorithm == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        # 可以扩展
        pass

    # 最小化损失
    train_step = optimizer.minimize(loss, global_step)
    return global_step, train_step, learning_rate

def add_loss_op(self, y_predict, y):
    '''
    添加损失函数操作
    :param self:当前模型的对象
    :param y_predict: y的预测值
    :param y: y的真实值
    :return: 当前损失值
    '''
    # 交叉熵
    cross_entropy = -tf.reduce_sum(tf.to_float(y) * tf.log(tf.clip_by_value(y_predict, 1e-20, 1.0)), 1)

    loss_sum = tf.reduce_sum(cross_entropy)

    return loss_sum

def add_loss_op_l2(self, y_predict, y, l2=None):
    '''
    添加损失函数操作
    :param self:当前模型的对象
    :param y_predict: y的预测值
    :param y: y的真实值
    :param l2: l2正则化的系数
    :return: 当前损失值
    '''

    # 交叉熵   tf.clip_by_value(y_predict, 1e-20, 1.0) 防止0出现导致nan
    # cross_entropy = -tf.reduce_sum(tf.to_float(y) * tf.log(y_predict), 1)
    cross_entropy = -tf.reduce_sum(tf.to_float(y) * tf.log(tf.clip_by_value(y_predict, 1e-20, 1.0)), 1)

    # 取这一批的均值
    loss_mean = tf.reduce_mean(cross_entropy)

    # 取这一批的总和
    loss_sum = tf.reduce_sum(cross_entropy)

    # 是否使用l2正则
    if l2 == None:
        # return loss_mean
        return loss_sum
    else:
        # 加入正则化防止过拟合
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding])
        # loss = loss_mean + l2 * reg_loss
        loss = loss_sum + l2 * reg_loss
        return loss

def gather_last_output(outputs, sent_len_ph):
    '''
    句长的最后一个lstm节点输出作为句子向量表示(效果比取平均好)
    :param outputs: rnn输出
    :param sent_len_ph: 每句长度
    :return: 取出每批的神经网络最后一个词对应的输出
    '''
    # outputs_deal = outputs[:,self.sent_len_ph,:]
    # 生成[0, 1, 2, 3...]对应批数
    id = tf.range(tf.shape(sent_len_ph)[0])
    # 转换为2D
    id_2D = tf.reshape(id, [-1, 1])
    # sent_len的数需要减1,因为数组下标从0开始
    sent_len_2D = tf.reshape(sent_len_ph - 1, [-1, 1])
    ids = tf.concat([id_2D, sent_len_2D], 1)
    # 取出每批的神经网络最后一个词对应的输出   shape(ids)=[batch,2] 例:[[0,4],[2,7],[3,3]...]
    return tf.gather_nd(outputs, ids)