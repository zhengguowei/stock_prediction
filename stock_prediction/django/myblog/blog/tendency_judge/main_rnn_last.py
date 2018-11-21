# coding=utf-8
import sys
import util1
import tf_util
import tensorflow as tf
from sklearn.metrics import classification_report
import time
import os
import argparse
reload(sys)
sys.setdefaultencoding('utf-8')

FLAGS = None

def add_arguments(parser):
    path=os.getcwd()
    '''
    添加参数解析
    :param parser:argparse.ArgumentParser()
    '''
    parser.add_argument("-schema", type=int, default=2, help='0:训练 1：测试 2：预测')
    parser.add_argument("-lr", type=float, default=0.01, help='learning rate')
    parser.add_argument("-decay_rate", type=float, default=0.9, help='学习率的衰减率')
    parser.add_argument("-max_epoch", type=int, default=20)
    parser.add_argument("-early_stop", type=int, default=3)
    parser.add_argument("-num_layer", type=int, default=1)
    parser.add_argument("-embedding_dim", type=int, default=200)
    parser.add_argument("-limit_sentence", type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-hidden_size', type=int, default=200)
    parser.add_argument("-target_file", type=str, default=path+'/blog/tendency_judge/model/tag.txt')
    parser.add_argument('-vocab_file', type=str, default=path+'/blog/tendency_judge/model/vocab.txt')
    parser.add_argument("-train_file", type=str, default='data/train.txt')
    parser.add_argument('-dev_file', type=str, default='data/dev.txt')
    parser.add_argument('-test_file', type=str, default='data/test.txt')
    # parser.add_argument('-predict_file', type=str, default='data/predict_src.txt')
    parser.add_argument('-predict_file', type=str, default=path+'/blog/tendency_judge/data/p_p.xls')
    parser.add_argument('-neg_file', type=str, default='model/neg_dic.txt')
    parser.add_argument('-neg_extent_dim', type=int, default=8)
    parser.add_argument('-embedding_file', type=str, default='model/embeddings.200')
    parser.add_argument('-model_path', type=str, default=path+'/blog/tendency_judge/model/rnn_model_1222/')

class Config(object):
    '''各种参数'''
    def __init__(self, FLAGS):
        print('开始加载词典...')
        self.vocab = util1.Vocab()
        self.vocab.load_vocab_from_file(FLAGS.vocab_file)
        print('开始加载标签...')
        self.tag_vocab = util1.Tag()
        self.tag_vocab.load_tag_from_file(FLAGS.target_file)
        self.limit_sentence = FLAGS.limit_sentence
        self.model_path = FLAGS.model_path

        if FLAGS.schema == 0:
            print('**************************\n初始化训练参数')
            self.lr = FLAGS.lr
            self.decay_rate = FLAGS.decay_rate
            self.max_epoch = FLAGS.max_epoch
            self.early_stop = FLAGS.early_stop
            self.num_layer = FLAGS.num_layer
            self.batch_size = FLAGS.batch_size
            self.hidden_size = FLAGS.hidden_size

            print('加载负面词典')
            self.neg_vocab = util1.load_neg_file(FLAGS.neg_file)

            print('开始读取数据...')
            # self.data_x, self.data_y, self.sent_len = util.read_data(FLAGS.train_file, self.tag_vocab, self.vocab, self.limit_sentence)
            self.data_x, self.data_y, self.sent_len = util1.read_data_seg(FLAGS.train_file, self.tag_vocab, self.vocab, self.limit_sentence)
            print('数据读取完毕')

            print('读取测试集...')
            # self.dev_x, self.dev_y, self.dev_sent_len = util.read_data(FLAGS.dev_file, self.tag_vocab, self.vocab, self.limit_sentence)
            self.dev_x, self.dev_y, self.dev_sent_len = util1.read_data_seg(FLAGS.dev_file, self.tag_vocab, self.vocab, self.limit_sentence)
            print('数据读取完毕')

            print('加载embedding文件')
            self.embedding_matrix = util1.create_embed_matrix_from_file(FLAGS.embedding_file, self.vocab)
            self.embedding_size = FLAGS.embedding_dim
            if FLAGS.neg_extent_dim > 0:
                # 在词向量中添加负面词
                print('embedding扩展')
                self.embedding_matrix = util1.extend_dim(self.embedding_matrix, self.vocab, self.neg_vocab, FLAGS.neg_extent_dim, 0.5)
                self.embedding_size += FLAGS.neg_extent_dim

            # 学习率衰减步
            self.decay_step = len(self.sent_len) / self.batch_size
        elif FLAGS.schema == 1:
            self.test_file = FLAGS.test_file
            pass
        elif FLAGS.schema == 2:
            # txt文本输入
            # self.predict_file = FLAGS.predict_file

            # excel文本输入
            path = os.getcwd()
            path = path+'/blog/tendency_judge/data/predict_src.txt'
            util1.extract_excel(FLAGS.predict_file, path)
            self.predict_file = path
            pass
        else:
            print('模式选择错误，0：训练 1：测试 2：预测')
    pass

class model():
    def __init__(self, sess):
        self.session = sess
        # 获取模型的节点
        self.x_ph = sess.graph.get_tensor_by_name('x_ph:0')
        self.y_ph = sess.graph.get_tensor_by_name('y_ph:0')
        self.sent_len_ph = sess.graph.get_tensor_by_name('sent_len_ph:0')
        self.y_predict = sess.graph.get_tensor_by_name('y_predict:0')

    def train(self, x_batch, y_batch, sent_len_batch):
        '''
        训练函数
        :param x_batch: x
        :param y_batch: y
        :param sent_len_batch:句子长度
        :return:
            loss ： 损失函数值
            step_now：总次数
            learning_rate_now：当前学习率
            accuracy_now：当前正确率
        '''
        _, loss, step_now, learning_rate_now, accuracy_now = self.session.run([self.train_step, self.loss, self.global_step, self.learning_rate, self.accuracy],
                                                                              feed_dict={self.x_ph: x_batch, self.y_ph: y_batch, self.sent_len_ph: sent_len_batch})

        return loss, step_now, learning_rate_now, accuracy_now

    def test(self, test_x, test_y, test_sent_len, tag_vocab):
        '''
        测试
        :param x_batch:
        :param y_batch:
        :param sent_len_batch:
        :return: 返回正确率
        '''
        result_true = util1.decode_tag(test_y, tag_vocab)
        result, probability = self.predict(test_x, test_sent_len, tag_vocab)

        # 打印结果
        print(classification_report(result_true, result))

        accuracy = util1.show_accuracy(result_true, result)
        return accuracy, result, probability


    def predict(self, x, sent_len, tag_vocab):
        '''
        预测
        :param x:
        :return:
        '''
        result = []
        probability = []

        for (x_batch, sent_len_batch) in util1.data_order_iter(x, sent_len, 2000):
            result_array = self.session.run(self.y_predict, feed_dict={self.x_ph:x_batch, self.sent_len_ph: sent_len_batch})
            probability.extend(result_array)
            result.extend(util1.decode_tag(result_array, tag_vocab))

        return result, probability

class RNN_model(model):
    def __init__(self, session, param_config):
        self.session = session
        CLASS_NUM = param_config.tag_vocab.total_tag
        HIDDEN_SIZE = param_config.hidden_size
        rnn_numLayers = param_config.num_layer

        # x,y placeholder
        self.x_ph = tf.placeholder(tf.int32, [None, param_config.limit_sentence], name='x_ph')
        self.y_ph = tf.placeholder(tf.int32, [None, CLASS_NUM], name='y_ph')
        self.sent_len_ph = tf.placeholder(tf.int32, [None], name='sent_len_ph')

        # 定义变量
        softmax_W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, CLASS_NUM], stddev=0.01))
        softmax_b = tf.Variable(tf.zeros([CLASS_NUM], tf.float32))

        # # x 添加embedding
        self.embedding, x_ph_embedding = tf_util.add_embedding(param_config.embedding_matrix, self.x_ph)

        # rnn
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        mul_lstm_cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell] * rnn_numLayers, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(mul_lstm_cell, x_ph_embedding, sequence_length=self.sent_len_ph, dtype=tf.float32, swap_memory=True)

        outputs_deal = tf_util.gather_last_output(outputs, self.sent_len_ph)
        # 取平均
        # outputs_deal = tf_util.data_mean(outputs, self.sent_len_ph, 1)

        # 预测
        self.y_predict = tf.nn.softmax(tf.matmul(outputs_deal, softmax_W) + softmax_b, name='y_predict')

        # 损失函数
        self.loss = tf_util.add_loss_op(self, self.y_predict, self.y_ph)

        # 精确度
        self.accuracy = tf_util.check_accuracy(self.y_ph, self.y_predict)

        # 随机梯度下降
        self.global_step, self.train_step, self.learning_rate = tf_util.train_optimizer(loss=self.loss,
                                                                                        learning_rate_start=param_config.lr,
                                                                                        decay_steps=param_config.decay_step,
                                                                                        decay_rate=param_config.decay_rate,
                                                                                        algorithm='SGD')
        # 全局初始化
        init = tf.global_variables_initializer()
        self.session.run(init)
    pass

def train_model(param_config):
    '''
    训练主函数
    '''
    # 创建文件夹
    save_path = param_config.model_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session() as sess:
        # 创建模型
        model = RNN_model(sess, param_config)
        # 获取保存对象
        saver = tf.train.Saver()
        # 记录最高正确率
        best_accuracy = 0.0
        # 记录每次的正确率
        each_accuracy = 0.0
        # 记录最好正确率对应的epoch
        best_epoch = 0

        # 开始循环
        for epoch in range(param_config.max_epoch):
            print('epoch：{}'.format(epoch))
            # 分批训练
            for step, (x_batch, y_batch, sent_len_batch) in enumerate(util1.data_iter(param_config.data_x, param_config.data_y, param_config.sent_len, param_config.batch_size)):

                # 训练
                loss, step_now, learning_rate_now, accuracy_now = model.train(x_batch, y_batch, sent_len_batch)

                if (step+1) % 500 == 0:
                    print('第{}批,loss：{},当前导入总批数：{},当前学习率:{},正确率：{}'.format(step+1, loss, step_now, learning_rate_now, accuracy_now))

            # 测试
            each_accuracy, _, _ = model.test(param_config.dev_x, param_config.dev_y, param_config.dev_sent_len, param_config.tag_vocab)
            print('开发集正确率:{}'.format(each_accuracy))

            # 记录最好的模型
            if best_accuracy < each_accuracy:
                best_accuracy = each_accuracy
                best_epoch = epoch
                # 模型保存
                saver.save(sess, '{}model.weights'.format(save_path))
            '''
            # 保存每个epoch的模型
            path = save_path+'{}/'.format(epoch)
            os.makedirs(path)
            saver.save(sess, '{}model.weights'.format(path))
            '''

            # 如果EARLY_STOPPING次后正确率平没有提升，则提前退出
            if epoch - best_epoch > param_config.early_stop:
                print("Normal Early stop")
                break
    pass

def t_est_new(param_config):
    '''
    预测（现在其实是测试）
    :param model_path:模型的路径
    :param file_path:要预测的文件
    :return:预测值
    '''
    model_path = param_config.model_path
    file_path = param_config.test_file

    # 读取文件
    # test_x, test_y, test_sent_len = util.read_data(file_path, param_config.tag_vocab, param_config.vocab, param_config.limit_sentence)
    test_x, test_y, test_sent_len = util1.read_data_seg(file_path, param_config.tag_vocab, param_config.vocab, param_config.limit_sentence)

    # 保存解码结果
    result = []
    probability = None

    with tf.Session() as sess:
        # 加载图模型
        saver = tf.train.import_meta_graph(model_path+'model.weights.meta')
        classifier = model(sess)
        # 加载参数
        saver.restore(sess, model_path+'model.weights')

        _, result, probability = classifier.test(test_x, test_y, test_sent_len, param_config.tag_vocab)

    # print result
    # 保存预测结果
    result_true = util1.decode_tag(test_y, param_config.tag_vocab)
    with open('result.txt', 'w', encoding='utf-8') as f:
        for (pre, true, prob) in zip(result, result_true, probability):
            f.write('{}\t{}\t{}\n'.format(pre, true, list(prob)))


def predict(param_config):
    '''
    预测
    :param model_path:模型的路径
    :param file_path:要预测的文件
    :return:预测值
    '''
    model_path = param_config.model_path
    file_path = param_config.predict_file

    # 读取文件
    # x, sent_len = util.read_predict_data(file_path, param_config.vocab, param_config.limit_sentence)
    x, sent_len = util1.read_predict_data_seg(file_path, param_config.vocab, param_config.limit_sentence)

    # 保存解码结果
    result = []
    probability = []

    with tf.Session() as sess:
        # 取回模型及参数
        saver = tf.train.import_meta_graph(model_path+'model.weights.meta')  # 返回 tf.train.Saver()
        classifier = model(sess)
        saver.restore(sess, model_path+'model.weights')

        r, p = classifier.predict(x, sent_len, param_config.tag_vocab)
        result.extend(r)
        probability.extend(p)

    # 输出预测文件

    # 保存成txt格式  标签\t[中立概率, 负面概率]
    path = os.getcwd()
    with open(path+'/blog/tendency_judge/predict.txt', 'w') as f:
        for (pre, prob) in zip(result, probability):
            f.write('{}\t{}\n'.format(pre, list(prob)))

    # 输出为excel
    path=os.getcwd()
    excel_result = []
    predict_src_list = []
    with open(path+'/blog/tendency_judge/data/predict_src.txt', 'r') as f:
        for line in f:
            sent = line.strip()
            if sent == '中立':
                sent = ''
            predict_src_list.append(sent)

    for (sent, pre, prob) in zip(predict_src_list, result, probability):
        excel_result.append([sent, pre, str(list(prob)[0]), str(list(prob)[1])])

    util1.excel_write(excel_result)


def main(_):
    start_time = time.time()
    param_config = Config(FLAGS)
    if FLAGS.schema == 0:
        train_model(param_config)
        pass
    elif FLAGS.schema == 1:
        t_est_new(param_config)
        pass
    elif FLAGS.schema == 2:
        predict(param_config)
    pass

    end_time = time.time()
    print('用时：{}'.format(end_time-start_time))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    FLAGS = arg_parser.parse_args()
    tf.app.run()  # 使用flags处理命令行指令,然后执行main（gpf）