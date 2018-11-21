# coding=utf-8

import sys
import os
import numpy as np
import jieba
import xlrd
import xlwt
import re
import time
import html
reload(sys)
sys.setdefaultencoding('utf-8')

'''
本模块作为辅助工具
    class：
        Vocab                            --词表
            load_vocab_from_file         --从词表文件中初始化词表
            encode                       --返回词汇在词汇表中的索引
            decode                       --根据索引从词汇表中查词
        Tag                              --标签
            load_tag_from_file           --从标签文件中初始化标签
            encode                       --返回one-hot形式的编码
            decode                       --根据编码返回标签

    function：
        create_embed_matrix_from_file    --从embedding文件创建词向量矩阵
            read_word_embeddings         --读入文件返回词典
            create_embed_matrix          --构建向量矩阵
        read_data                        --读入数据(一次存入内存)
        read_data_seg                    --读数据分词
        sentence_encode                  --句子编码表示
        data_iter                        --数据迭代器（随机返回）
        data_order_iter                  --数据迭代器（按顺序返回）
        data_mean                        --使用numpy的接口，处理词取平均的方式表示句子
        read_predict_data                --加载预测数据
        read_predict_data_seg            --加载未分词的测试数据
        clac_accuracy                    --计算正确率
        decode_tag                       --标签解码
        show_accuracy                    --统计正确率（参数是两个列表）
        load_neg_file                    --加载负面词典
        extend_dim                       --扩展词向量
        extrect_excel                    --从excel中提取出文本数据
        excel_write                      --预测结果写出到excel
        segment                          --分词
        strQ2B                           --符号转换
        str_re                           --去[]
        neg_first                        --提负面词 
        
'''

class Vocab():
    '''
    词表
    '''

    def __init__(self, unk='<unk>'):
        '''
        构造函数
        :param unk: 词表以外的词的表示形式
        '''
        self.word_to_index = {}  # word:index
        self.index_to_word = {}  # index:word
        self.total_word = 0  # 总词数
        self.unknown = unk
        self.__add_word(self.unknown)

    def __add_word(self, word):
        '''
        加入词表
        :param word: 词
        :return: None
        '''
        index = len(self.word_to_index)
        self.word_to_index[word] = index
        self.index_to_word[index] = word

    def load_vocab_from_file(self, vocab_file):
        '''
        从词表文件中初始化词表
        :param vocab_file: 文件路径
        :return: None
        '''
        if os.path.isfile(vocab_file) and os.path.exists(vocab_file):
            with open(vocab_file, 'rb') as f:
                for line in f:
                    line = line.decode('utf-8').strip()
                    word_count = line.split('\t')
                    word = word_count[0]
                    self.__add_word(word)

            self.total_word = len(self.word_to_index)
            print('词表中有{}个词'.format(self.total_word - 1))
        else:
            raise ValueError('文件路径有误！！')
        pass

    def encode(self, word):
        '''
        返回词汇在词汇表中的索引,当字符不在词汇表中时用unk表示
        :param word: 词
        :return:
        '''
        # 词不在词表中用unknown表示
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        '''
        根据索引从词汇表中查词
        :param index:
        :return:
        '''
        return self.index_to_word[index]

    def print_vocab(self):
        for (id, word) in self.index_to_word.items():
            print('{}\t{}'.format(id, word))


class Tag():
    '''
    标签表
    '''

    def __init__(self):
        '''
        构造函数
        :param unknown: 标签表以外的类型的表示形式
        '''
        self.tag_to_index = {}
        self.index_to_tag = {}
        self.total_tag = 0

    def __add_tag(self, tag):
        '''
        添加标签
        :param
            tag: 标签
        :return:Nne
        '''
        index = len(self.tag_to_index)
        self.tag_to_index[tag] = index
        self.index_to_tag[index] = tag

    def load_tag_from_file(self, tag_file):
        '''
        从标签文件中初始化标签
        :param
            tag_file: 文件路径
        :return: None
        '''
        if os.path.isfile(tag_file) and os.path.exists(tag_file):
            with open(tag_file, 'rb') as f:
                for tag in f:
                    tag = tag.decode('utf-8').strip()
                    self.__add_tag(tag)

            # 计算标签总个数
            self.total_tag = len(self.tag_to_index)
            print('标签表中有{}个类别'.format(self.total_tag))
        else:
            raise ValueError('文件路径有误！！')
        pass

    def encode(self, tag):
        '''
        返回one-hot形式的编码
        :param tag: 标签
        :return:标签向量  [0 0 0 1 0]
        '''
        tag_vec = np.array([0] * self.total_tag)
        tag_vec[self.tag_to_index[tag]] = 1
        return tag_vec

    def decode(self, tag_vec):
        '''
        解码
        :param tag_vec:标签的向量表示
        :return: 标签
        '''
        index = np.argmax(tag_vec)
        return self.index_to_tag[index]

    def print_tag(self):
        for (id, word) in self.index_to_tag.items():
            print('{}\t{}'.format(id, word))


def create_embed_matrix_from_file(embeddings_file, vocab):
    '''
    从embedding文件创建词向量矩阵
    :param embeddings_file: embedding文件
    :param vocab:词典
    :return:词向量矩阵
    '''

    def read_word_embeddings(embeddings_file):
        '''
        读入词向量文件,返回词典
        :param
            embeddings_file ：文件路径
        :return:a dictionary contains the mapping from word to vector
        '''
        # 存储词向量
        word_embeddings = {}
        if os.path.isfile(embeddings_file) and os.path.exists(embeddings_file):
            print('开始加载embedding文件...')
            with open(embeddings_file, 'rb') as f:
                i = 0
                for line in f:
                    line = line.decode('utf-8').strip()
                    values = line.split()
                    word = values[0]
                    embedding = np.array(values[1:], dtype='float32')
                    word_embeddings[word] = embedding
                    i = i + 1
                    if i % 100000 == 0:
                        print('加载{}'.format(i))
            print('共加载{}个词向量'.format(i))
            print('embedding文件加载完毕')
            return word_embeddings
        else:
            raise ValueError('文件路径有误！！')
        pass

    def create_embed_matrix(embeddings_dic, vocab_dic):
        '''
        根据词向量词典和词表构建词向量矩阵
        :param embeddings_dic: word-vectors 词典
        :param vocab_dic: word-index 词典
        :return:词的向量矩阵，每行表示一个词
        '''
        if type(embeddings_dic) is not dict or type(vocab_dic) is not dict:
            raise TypeError('Inputs are not dictionary')
        # 确定维度大小
        embedding_size = len(embeddings_dic[list(embeddings_dic.keys())[0]])
        # 申请矩阵空间
        word_mun = len(vocab_dic)
        embeddings_matrix = np.zeros((word_mun, embedding_size), dtype=np.float32)

        # 填充矩阵
        for (word, index) in vocab_dic.items():
            vector = embeddings_dic.get(word)
            # 如果在embedding词典中没有对应的词，则全零
            if vector is not None:
                embeddings_matrix[index] = vector
        return embeddings_matrix

    # 获取embedding词典
    embeddings_dic = read_word_embeddings(embeddings_file)
    # 获取词表词典
    vocab_dic = vocab.word_to_index
    # 构建词向量矩阵
    return create_embed_matrix(embeddings_dic, vocab_dic)
    pass


def read_data(file, tag_vocab, vocab, limit):
    '''
    读入数据(选择全部读入内存的方式)
    按行存储，类别\t已经分好词（用空格隔开）
    如：
        类   我 爱 中国
    :param
        file：文件路径
        tag_vocab：标签表
        vocab：词表
        limit:句子限制长度
    :return:DataSet 转换为词表中坐标 ，三个都是数组形式
    '''
    x_set = []  # 存句子
    y_set = []  # 存标签
    sent_len = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tag_words = line.strip().split()
            # 得到标签的编码
            y_set.append(tag_vocab.encode(tag_words[0]))
            # 得到句子编码和句子长度
            sent_vec, sent_length = sentence_encode(tag_words[1:], vocab, limit)
            x_set.append(sent_vec)
            sent_len.append(sent_length)
    return np.array(x_set), np.array(y_set), np.array(sent_len)

def read_data_seg(file, tag_vocab, vocab, limit):
    '''
    读入数据(选择全部读入内存的方式)
    按行存储，未分词
    如：
        类   我爱中国
    :param
        file：文件路径
        tag_vocab：标签表
        vocab：词表
        limit:句子限制长度
    :return:DataSet 转换为词表中坐标 ，三个都是数组形式
    '''
    jieba.load_userdict('model/user_dic.txt')
    x_set = []  # 存句子
    y_set = []  # 存标签
    sent_len = []
    neg_vocab = load_neg_file('model/extract_neg_dic.txt')
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = str_filter(line)
            tag_sent = line.split('\t')
            # 得到标签的编码
            y_set.append(tag_vocab.encode(tag_sent[0]))
            # 得到句子编码和句子长度
            if tag_sent[1].strip() == '':
                words = ['中立']
            else:
                # 分词
                words = segment(tag_sent[1])
            print(words)
            sent_vec, sent_length = sentence_encode(words, vocab, limit)
            x_set.append(sent_vec)
            sent_len.append(sent_length)
    return np.array(x_set), np.array(y_set), np.array(sent_len)

def read_data_seg_neg(file, tag_vocab, vocab, limit):
    '''
    读入数据(选择全部读入内存的方式)
    按行存储，未分词
    如：
        类   我爱中国
    :param
        file：文件路径
        tag_vocab：标签表
        vocab：词表
        limit:句子限制长度
    :return:DataSet 转换为词表中坐标 ，三个都是数组形式
    '''
    jieba.load_userdict('model/user_dic.txt')
    x_set = []  # 存句子
    y_set = []  # 存标签
    sent_len = []
    neg_vocab = load_neg_file('model/extract_neg_dic.txt')
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = str_filter(line)
            tag_sent = line.split('\t')
            # 得到标签的编码
            y_set.append(tag_vocab.encode(tag_sent[0]))
            # 得到句子编码和句子长度
            if tag_sent[1].strip() == '':
                words = ['中立']
            else:
                # 分词
                words = segment(tag_sent[1])
                # 提负面词
                words = neg_first(words, neg_vocab)
            print(words)
            sent_vec, sent_length = sentence_encode(words, vocab, limit)
            x_set.append(sent_vec)
            sent_len.append(sent_length)
    return np.array(x_set), np.array(y_set), np.array(sent_len)

def read_predict_data(file, vocab, limit):
    '''
    读入数据(选择全部读入内存的方式)
    按行存储，已经分好词（用空格隔开）
    如：
        类   我 爱 中国
    :param
        file：文件路径
        vocab：词表
        limit:句子限制长度
    :return:DataSet 转换为词表中坐标 ，三个都是数组形式
    '''
    x_set = []  # 存句子
    sent_len = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line.strip()
            line = str_filter(line)
            words = line.split()  # 会以 ' '隔开
            # print(words)
            # 得到句子编码和句子长度
            sent_vec, sent_length = sentence_encode(words, vocab, limit)
            x_set.append(sent_vec)
            sent_len.append(sent_length)
    return np.array(x_set), np.array(sent_len)

def read_predict_data_seg_neg(file, vocab, limit):
    '''
    读入数据(选择全部读入内存的方式)
    按行存储，未分词
    如：
        我爱中国
    :param
        file：文件路径
        vocab：词表
        limit:句子限制长度
    :return:DataSet 转换为词表中坐标 ，三个都是数组形式
    '''
    jieba.load_userdict('model/user_dic.txt')
    x_set = []  # 存句子
    sent_len = []
    neg_vocab = load_neg_file('model/extract_neg_dic.txt')

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = str_filter(line)

            if line.strip() == '':
                words = ['中立']
            else:
                # 分词
                words = segment(line)

                # 提负面词
                words = neg_first(words, neg_vocab)

            print(words)
            # 得到句子编码和句子长度
            sent_vec, sent_length = sentence_encode(words, vocab, limit)
            x_set.append(sent_vec)
            sent_len.append(sent_length)
    return np.array(x_set), np.array(sent_len)

def read_predict_data_seg(file, vocab, limit):
    '''
    读入数据(选择全部读入内存的方式)
    按行存储，未分词
    如：
        我爱中国
    :param
        file：文件路径
        vocab：词表
        limit:句子限制长度
    :return:DataSet 转换为词表中坐标 ，三个都是数组形式
    '''
    path=os.getcwd()
    jieba.load_userdict(path+'/blog/tendency_judge/model/user_dic.txt')
    x_set = []  # 存句子
    sent_len = []
    with open(file, 'r') as f:
        for line in f:
            line.strip()
            line = str_filter(line)
            words = segment(line)
            # print(words)
            # 得到句子编码和句子长度
            sent_vec, sent_length = sentence_encode(words, vocab, limit)
            x_set.append(sent_vec)
            sent_len.append(sent_length)
    return np.array(x_set), np.array(sent_len)

def sentence_encode(words, vocab, limit=0):
    '''
    句子编码
    :param words:句子中的词
    :param vocab:词表
    :param limit:句子的限制词数 (如果为0，表示不限制长度)
    :return:句子的标记表示,句子长度
    '''
    sent_vec = []  # 句子表示
    length = len(words)  # 句子长度
    # 是否截断句子
    if limit == 0:
        for word in words:
            sent_vec.append(vocab.encode(word))

    else:
        length = min(limit, length)
        for i in range(limit):
            if i < length:
                sent_vec.append(vocab.encode(words[i]))
            else:
                # 用unknown补齐
                sent_vec.append(vocab.encode(vocab.unknown))
    return np.array(sent_vec), length
    pass


def data_iter(data_x, data_y, len_arr, batch_size):
    '''
    数据迭代器(返回随机的数据)
    :param data_x:数据内容
    :param data_y:标签
    :param len_list:句子长度集合
    :param batch_size:批处理大小
    :return: 数据矩阵： x,y,句子长度
    '''
    # 数据总量
    data_len = len(len_arr)

    total_iter = data_len // batch_size

    for _ in range(total_iter):
        indexes = np.random.choice(data_len, batch_size)
        yield (data_x[indexes], data_y[indexes], len_arr[indexes])
    pass

def data_order_iter(data_x, len_arr, batch_size):
    '''
    数据迭代器（数据按顺序返回）
    :param data_x:数据内容
    :param len_list:句子长度集合
    :param batch_size:批处理大小
    :return: 数据矩阵： x,y,句子长度
    '''
    # 数据总量
    data_len = len(len_arr)

    total_iter = data_len // batch_size

    for i in range(total_iter+1):
        indexes = np.arange(i*batch_size, (i+1)*batch_size, 1)  # 生成连续的batch_size个数
        try:
            # 由于数组长度不一定整出batch_size,所以可能会数组越界
            yield (data_x[indexes], len_arr[indexes])
        except:
            yield (data_x[i*batch_size:], len_arr[i*batch_size:])
    pass


def data_mean(x_batch, sent_len_batch):
    '''
    算句子的向量表示，用词算平均值的方式
    :param x_batch:一批句子表示
    :param sent_len_batch: 句子长度
    :return: 一批句子向量
    '''
    size = len(x_batch)
    mean_batch = []
    for i in range(size):
        word_list = []
        # 去掉句子长度以后的 用0填塞的部分
        for j in range(sent_len_batch[i]):
            word_list.append(x_batch[i][j])

        # 句内平均
        mean_batch.append(np.mean(np.array(word_list), 0))
    return np.array(mean_batch)


def clac_accuracy(y_pre, y_true):
    '''
    :param y_pre: 预测数据标签
    计算正确率
    :param y_true: 真实数据标签
    :return:
    '''
    return np.mean(np.equal(np.argmax(y_pre, 1), np.argmax(y_true, 1)))


def decode_tag(result_array, tag):
    '''
    预测结果解码
    :param result_array:标签矩阵
    :return:返回结果列表
    '''
    result = []
    for vector in result_array:
        result.append(tag.decode(vector))
    return result

def load_neg_file(file):
    '''
    加载负面词
    :param file:负面词词典
    :return:list
    '''
    neg_vocab = []
    with open(file, 'r', encoding='utf-8') as n_f:
        for word in n_f:
            neg_vocab.append(word.strip())
    return neg_vocab

def show_accuracy(y_true, y_pre):
    '''
    比较两个列表数据之间的正确率
    :param y_true: 真实数据
    :param y_pre: 测试数据
    :return:总体的正确率
    '''
    acc = np.array(y_true) == np.array(y_pre)
    return np.mean(acc)

def extend_dim(embedding, vocab, negative_vocab, ex_dim=1, init_value=1):
    '''
    在词上加入情感信息
    :param embedding: 词向量矩阵
    :param vocab: 词典（dict）
    :param negative_vocab:负面词典（list） 后期修改可以使用dict，value不同词对应不同的值
    :param init_value:负面词初始值
    :return:扩展后的词向量
    '''
    neg_matrix = np.zeros([vocab.total_word, ex_dim], dtype=np.float32)
    for word in negative_vocab:
        neg_matrix[vocab.word_to_index[word]] = np.array([init_value]*ex_dim)
    return np.column_stack((embedding, neg_matrix))

def extract_excel(excel, output):
    '''
    第一行 表头不读
    第一列 行号不读
    只读第二列数据
    空行用‘中立’替换
    :param excel:excel文件
    :param output:输出提取内容文件路径
    :return:
    '''
    print('开始提取'+excel+'中数据')
    # 打开excel
    data = xlrd.open_workbook(excel)
    # 获取工作表
    table = data.sheets()[0]
    nrows = table.nrows # 行数
    # ncols = table.ncols # 列数

    f = open(output, 'w')

    # 跳过表头
    for row_id in range(1, nrows):
        row = table.row_values(row_id)
        sent = str(row[1])
        if sent == '':
            sent = '中立'
        f.write(sent+'\n')

    f.close()

def excel_write(result):
    '''
    将结果写出到excel
    :param result: [原句, 预测标签， 中立概率， 负面概率]
    :return:
    '''
    wbk = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = wbk.add_sheet('sheet 1', cell_overwrite_ok=True)  ##第二参数用于确认同一个cell单元是否可以重设值。

    for (r, list) in enumerate(result):
        for (c, value) in enumerate(list):
            if c == 2 or c == 3:
                sheet.write(r, c, float(value))
            else:
                sheet.write(r, c, value)

    wbk.save('predict.xls')

def segment(line):
    '''分词 返回列表'''
    # [x for x in jieba.cut(tag_sent[1]) if x != ' ' and x in vocab.word_to_index]
    return [x for x in jieba.cut(line.replace(' ','').strip(), cut_all=False)]

def strQ2B(ustring):
    """全角转半角"""
    rstring = ustring.replace('，', ',').replace('‘', '\'').replace('’', ',').replace('；', ';') \
            .replace('：', ':').replace('“', '"').replace('”', '"').replace('？', '?') \
            .replace('｜', '|').replace('！', '!').replace('【', '[').replace('】', ']') \
            .replace('『', '[').replace('』', ']').replace('「', '[').replace('」', ']')
    return rstring

def str_re(line):
    '''去除 []'''
    lab = '\[[\u4e00-\u9fa5]{0,20}\]'
    line_r = re.sub(lab, '', line)
    return line_r

def str_filter(line_h):
    '''句子清洗'''
#    line_h = replace_html(line)
    line_2 = strQ2B(line_h)
    line_r = str_re(line_2)
    return line_r

def neg_first(words, neg_vocab):
    '''将负面词典提前'''
    # 负面词典中出现的次提前
    for word in words:
        if word in neg_vocab:
            words.insert(0, word)
            break

    return words

def replace_html(line):
    return html.unescape(line)


if __name__ == '__main__':
    s = time.time()
    line = '【日本房产投资科普:价格、回报率、面积计算方式等等】'
    line = '&ldquo;金融知识万里行&rdquo;&mdash;&mdash;防范电信网络诈骗宣传月'
    print(replace_html(line))
    print(time.time()-s)
