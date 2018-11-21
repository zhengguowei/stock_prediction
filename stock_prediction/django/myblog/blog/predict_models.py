#-*- coding: utf-8 -*-
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from bs4 import BeautifulSoup
import urllib2
import datetime
import xlwt
import requests
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing
from datetime import datetime
from datetime import timedelta
from tendency_judge.main_rnn_last import *
import math
import random
import jieba

__author__='Zheng guowei'

id_to_name={}
name_to_id_short={}
name_to_id_full={}
path=os.getcwd()
fi=open(path+'/blog/dict/stock_dic1.txt','r')
lines=fi.readlines()
fi.close()
for i in range(len(lines)):
    lines[i]=lines[i].split(' ')
    id_to_name[lines[i][1]]=lines[i][2]
    name_to_id_short[lines[i][2]]=lines[i][1]
    name_to_id_full[lines[i][3]]=lines[i][1]

class Models:
    ss_x = preprocessing.StandardScaler()
    ss_y = preprocessing.StandardScaler()
    def __init__(self,ss_x=ss_x,ss_y=ss_y,id_to_name=id_to_name,name_to_id_short=name_to_id_short,name_to_id_full=name_to_id_full):
        self.ss_x=ss_x
        self.ss_y=ss_y
        self.id_to_name=id_to_name
        self.name_to_id_short=name_to_id_short
        self.name_to_id_full=name_to_id_full
#crawl page
    def download_stock(self,s):
        def get_content(url):
            content=None
            try:
                content = urllib2.urlopen(url,timeout=2).read()
            except:
                print 'get content failed'
            return content

#parse the content of web page
        def get_market_index(content):
            market_idnex_list = []
            content = BeautifulSoup(content, 'html.parser')
            try:
                table = content.find('table', id='FundHoldSharesTable').contents
                for i in range(5, len(table)):
                    if i % 2 != 0:
                        text = table[i].contents
                        temporary_text = []
                        for j in range(1, len(text)):
                            if j % 2 != 0:
                                temporary_text.append(text[j].text.strip())
                        market_idnex_list.append(temporary_text)
            except:
                pass
            return market_idnex_list

#write data to txt file
        def data_to_txt(market_index_list):
            fou = open('blog/tmp_file/history.txt', 'w')
            for line in market_index_list:
                fou.write(' '.join(line) + '\n')
            fou.close()

        index_url = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/%s/.phtml?year=20&jidu=2'%s
        market_idnex=[]
        for i in range(14, 19):
            for j in range(1, 5):
                if i == 7 and j >= 2:
                    continue
                if len(str(i)) == 1:
                    real_url = index_url[:96] + '0' + str(i) + index_url[96:102] + str(j)
                else:
                    real_url = index_url[:96] + str(i) + index_url[96:102] + str(j)
                print real_url
                content = get_content(real_url)
                if content:
                    market_idnex+= get_market_index(content)[::-1]
        data_to_txt(market_idnex)

#preprocess data
    def preprocess_data(self):
        path=os.getcwd()
        data=[]
        lines=open(path+'/blog/tmp_file/history.txt').readlines()
        for i in range(len(lines)):
            lines[i]=lines[i].split(' ')
            data.append(lines[i][0])
            lines[i]=map(float,lines[i][1:-2])
        train, target = [], []
        for j in range(len(lines)-30):
            tmp=[]
            for i in range(j,j+30):
                tmp+=lines[i]
            train.append(np.array(tmp))
            target.append(lines[j+30][2])
        remain_predict=[]
        five_test,five_true=[],[]
        for k in range(j+1,j+31):
            remain_predict+=lines[k]
        for i in range(len(lines)-35,len(lines)-30):
            tmp=[]
            for v in range(i,i+30):
                tmp+=lines[v]
            five_test.append(np.array(tmp))
            five_true.append(lines[i+30][2])
        for i in range(-5,0):
            data[i]=data[i][data[i].index('-')+1:]
        year_test,year_true,base=[],[],[]
        try:
            for i in range(len(lines)-275,len(lines)-30):
                tmp=[]
                for v in range(i,i+30):
                    tmp+=lines[v]
                year_test.append(np.array(tmp))
                year_true.append(lines[i+30][2])
                base.append(lines[i+29][2])
        except:
            pass
        return np.array(train),np.array(target),np.array(remain_predict),np.array(five_test),np.array(five_true),lines[-1][2],data[-5:],np.array(year_test),year_true,base

# bp model
    def bp_fit(self,x_train,y_train):
        # print 'the shape of x:', x_train.shape
        # print 'the shape of y:', y_train.shape
        print('begin fit')

#split the data
        x_train = self.ss_x.fit_transform(x_train)
        y_train = self.ss_y.fit_transform(y_train.reshape(-1, 1))

#  Multilayer perceptron
        model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(120, 9, 4), random_state=1)
        model_mlp.fit(x_train, y_train.ravel())
        model_gbr = GradientBoostingRegressor()
        model_gbr.fit(x_train, y_train.ravel())
        model_gbr_best = GradientBoostingRegressor(learning_rate=0.01, max_depth=6, max_features=0.5,
                                                   min_samples_leaf=14,
                                                   n_estimators=70)
        model_gbr_best.fit(x_train, y_train.ravel())
        return model_mlp,model_gbr

#bp model prediction
    def bp_predict(self,models,remain_pre):
        remain_pre = self.ss_x.transform(remain_pre)
        bp_predict=models[0].predict(remain_pre)
        fusion_predict=models[1].predict(remain_pre)
        bp_predict = self.ss_y.inverse_transform(bp_predict)
        fusion_predict = self.ss_y.inverse_transform(fusion_predict)
        return bp_predict,fusion_predict

# lstm model
    def lstm_fit(self,x_train,y_train):
        x_train=x_train.reshape(x_train.shape[0],120,1)
#        print x_train.shape,y_train.shape
        epochs=1
        layers=[1, 120, 10, 1]
        model = Sequential()
        model.add(LSTM(
            input_shape=(layers[1], layers[0]),
            output_dim=layers[1],
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[2],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=layers[3]))
        model.add(Activation("linear"))

        start = time.time()
        model.compile(loss="mse", optimizer="rmsprop")
        print("> Compilation Time : ", time.time() - start)
        model.fit(
            x_train,
            y_train,
            batch_size=128,
            nb_epoch=epochs,
            validation_split=0.05)
        return model

# juege whether param is a digit
    def judge_digit(self,num):
        try:
            int(num)
            return  True
        except:
            return False

#lstm prediction
    def lstm_predict(self,model,remain_predict):
        remain_predict=remain_predict.reshape(1,120,1)
        predicted = model.predict(remain_predict)
        predicted=self.ss_y.inverse_transform(predicted)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

#get company name according to company name
    def get_commpany_name(self,id):
        try:
            return self.id_to_name[id]
        except:

            return None

#get id according to id
    def get_company_id(self,name):
        if name in self.name_to_id_short:
            return self.name_to_id_short[name]
        # for x in self.name_to_id_full:
        #     if name in x:
        #             return self.name_to_id_full[name]
        return None
#get related news
    def get_related_news(self,name):
# download the content of the page
        def getYesterday():
            today=datetime.date.today()
            oneday=datetime.timedelta(days=1)
            yesterday=today-oneday
            return yesterday

        def get_content(url):
            try:
                content=requests.get(url)
                content.encoding='gb18030'
                return content.text

#                return urllib2.urlopen(url).read()
            except:
                return None

#get news related stock
        def get_title_list(content):
            titles = []
            content = BeautifulSoup(content, 'html.parser')
            tables = content.find_all('ul', class_='list_009')
            for i in range(len(tables)):
                title = tables[i].contents
                for i in range(len(title)):
                    if i % 2 != 0:
                        titles.append(title[i].text)
            return titles

#get urls of news
        def get_url_list(content):
            urls = []
            content = BeautifulSoup(content, 'html.parser')
            tables = content.find_all('ul', class_='list_009')
            for i in range(len(tables)):
                title = tables[i].contents
                for i in range(len(title)):
                    if i % 2 != 0:
                        tmp = title[i].contents[0].attrs['href']
                        urls.append(tmp)
            return urls

        data = time.strftime('%Y-%m-%d', time.localtime(time.time())) + '_'
        # data=getYesterday().strftime('%Y-%m-%d')+'_'
        title_list = []
        url_list = []
        result = set()
        for i in range(4):
            url = 'http://roll.finance.sina.com.cn/finance/zq1/ssgs/%s.shtml'
            real_url = url % (data + str(i))
            content = get_content(real_url)
            if content:
                tmp_titles = get_title_list(content)
                tmp_urls = get_url_list(content)
                url_list += tmp_urls
                title_list += tmp_titles
        if type(name)==list:
            name=name[0]
        name = name.decode('utf-8')
        for i in range(len(title_list)):
            if name in title_list[i]:
                result.add(title_list[i]+'***'+url_list[i])
        result=list(result)
        for i in range(len(result)):
            result[i]=result[i].split('***')
        import requests
        from bs4 import BeautifulSoup
        url='http://news.baidu.com/ns?cl=2&rn=20&tn=news&word=%s'%(name)
        content = requests.get(url)
        content=content.text
        content = BeautifulSoup(content, 'html.parser')
        content = content.find_all('div', class_='result')
        news = []
        date=[]
        for x in content:
            tmp = x.contents
            url = tmp[1].contents[1].attrs['href']
            title = x.text.split('\n')[3].strip()
            date1 = tmp[3].text.split('\n')[2].strip()
            if '小时' in date1 or '分钟' in date1:
                date.append(date1)
            news.append([title, url])
        return result+news[:len(date)]

#classify the news
    def get_news_tendency(self,content):
        path = os.getcwd()
        myWorkbook = xlwt.Workbook()
        mySheet = myWorkbook.add_sheet('A Test Sheet')
        mySheet.write(0,0,0)
        for i in range(1,len(content)+1):
            mySheet.write(i, 1,content[i-1][0])
        myWorkbook.save(path + '/blog/tendency_judge/data/p_p.xls')
        end='/blog/tendency_judge/main_rnn_last.py'
        path1=path
        path1+=end
        path1='python '+path1
#        os.system('python /Users/macbook/Desktop/web_test/django/myblog/blog/tendency_judge/main_rnn_last.py')
        os.system(path1)
        tendency1=open(path+'/blog/tendency_judge/predict.txt').readlines()
        for i in range(len(tendency1)):
            tendency1[i]=tendency1[i].split('\t')
        return tendency1
    def isexist(self,stock_id):
        return stock_id in self.id_to_name

    def downImg(self,imgUrl, dirpath, imgName):
        path = os.getcwd()
        dirpath=path+dirpath
        filename = os.path.join(dirpath, imgName)
        try:
            res = requests.get(imgUrl, timeout=15)
            if str(res.status_code)[0] == "4":
                print(str(res.status_code), ":", imgUrl)
                return False
        except Exception as e:
            print("raise ", imgUrl)
            print(e)
            return False
        with open(filename, "wb") as f:
            f.write(res.content)
        return True

#
    def stock_flag(self,stock_id):
        if stock_id in self.stock_flag:
            return self.stock_flag[stock_id]


    def draw_predictions(self,data,predictions):
        from pylab import plt
        true, lstm, bp = [], [], []
        for i in range(len(predictions)):
            true.append(predictions[i][0])
            bp.append(predictions[i][1])
            lstm.append(predictions[i][2])
        x = [1,2,3,4,5]
        plt.plot(x, true, 'cx--', label='true')
        plt.plot(x, bp, 'mo:', label='fusion')
        plt.plot(x, lstm, 'kp-.', label='lstm')
        plt.legend()
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"days")
        plt.ylabel("price")
        plt.title("tendency predictions of different models")
        plt.savefig('blog/static/blog/bootstrap/img/presult.jpg')


    def get_price(self,stock_flag,id):
        dic={'sh':'sz','sz':'sh'}
        base_rul='http://hq.sinajs.cn/list=%s%s'%(stock_flag,id)
        if len(urllib2.urlopen(base_rul).read())>60:
            tmp=urllib2.urlopen(base_rul).read()
            tmp=tmp.split(',')
            return tmp[3]
        else:
            base_rul = 'http://hq.sinajs.cn/list=%s%s' % (dic[stock_flag], id)
            tmp = urllib2.urlopen(base_rul)
            tmp = tmp.split(',')
            return tmp[3]

    def get_data(self,n):
        tmp = datetime.now() - timedelta(days=n)
        return str(tmp.month) + '-' + str(tmp.day)
    def get_result(self,bp,lstm,last_price,news):
        if bp>last_price and lstm>last_price:
            return '上涨的可能性较大'
        elif bp<last_price and lstm<last_price:
            return '下跌的可能性较大'
        else:
            tendency=[]
            if news:
                for x in news:
                    tendency.append(x[0])
                neg='负面' in tendency
                if neg:
                    return '下跌的可能性较大'
                else:
                    return '上涨的可能性较大'
        return '下跌的可能性较大' if random.uniform(0,1)>0.5 else '上涨的可能性较大'


    def sentences_extraction(self,text):
        text=text.split('。')
        text_lenght=len(text)
        def postion_score(postion,text_lenght):
            mid=float(text_lenght)/2
            return (postion-mid)**2+1
        def name_score(sentence):
            exist=False
            for x in self.name_to_id_short:
                if x in sentence:
                    exist=True
                    break
            return exist
        def  code_score(sentence):
            exist=False
            for x in self.name_to_id_short:
                if x in sentence:
                    exist=True
                    break
            return exist
        def sim(sentece1,sentence2):
            return np.dot(sentence2,sentece1)/math.sqrt(sum((np.array(sentece1)-np.array(sentence2))**2))
        remain=postion_score(text)
        remain=sorted(remain,key=lambda x:x[1])
        return remain


if __name__=='__main__':
    models=Models()
    for x in models.sentences_extraction('我曾经穿过时光隧道。然后去了太空'):
        print x