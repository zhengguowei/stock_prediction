# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
import time
from . import models
from predict_models import Models

__author__='Zheng guowei'

def index(request):
    return render(request,'test.html',{'hello':'hello world'})
def get_stock_id(request):
    model = Models()
    #get the id of a stock
    param=request.POST.get('stock_id','STOCK_ID')
    start_time=request.POST.get('start_time','start_time')
    end_time = request.POST.get('end_time', 'end_time')
    print start_time,end_time
    param=param.strip().encode('utf-8')
    flag=model.judge_digit(param)
    if flag:
        stock_id=param
    else:
        stock_id=model.get_company_id(param)
    isexist = model.isexist(stock_id)
    five=None
    last_price=''
    data=None
    lstm_benifit,fusion_benifit=0.0,0.0
    if isexist:
    #download history data of stock
        model.download_stock(stock_id)
        data_download=False
        stock_flag=''
        if int(stock_id)>300741:
            stock_flag='sh'
        else:
            stock_flag='sz'
        tmp_url='http://image.sinajs.cn/newchart/daily/n/%s%s.gif'%(stock_flag,stock_id)
        model.downImg(tmp_url,'/blog/static/blog/bootstrap/img','stock.gif')
    #preprocess the data
        try:
            train,target,remain_predict,five_test,five_true,last_price,data,year_test,year_true,base=model.preprocess_data()
        except:
            data_download=True
    #fit bp model
        if not data_download:
            fusion_model=model.bp_fit(train,target)
    # predict the bp value of the stock of next day
            bp_predict, fusion_predict=model.bp_predict(fusion_model,remain_predict)
            year_prediction=model.bp_predict(fusion_model,year_test)
            fusion_benifit=0.0
            lstm_benifit=0.0
            fusion_year_prediction,lstm_year_prediction=year_prediction[0].tolist(),year_prediction[1].tolist()

            for i in range(len(fusion_year_prediction)):
                if fusion_year_prediction[i]>base[i]:
                    fusion_benifit+=(year_true[i]-base[i])
                if lstm_year_prediction[i]>base[i]:
                    lstm_benifit+=(year_true[i]-base[i])
            #i=0
            # while i<len(fusion_year_prediction)-1:
            #     if fusion_year_prediction[i]>=base[i]:
            #         continue
            #     else:
            #         fusion_benifit += (fusion_year_prediction[i] - year_true[i])
            #     if lstm_year_prediction[i]>base[i]:
            #         lstm_benifit += (lstm_year_prediction[i] - year_true[i])
            #     i+=1

            five_bp,five_lstm=model.bp_predict(fusion_model,five_test)
            five_bp, five_lstm=five_bp.tolist(),five_lstm.tolist()
            five=[]
            predictions=[]
            for i in range(len(five_true)):
                predictions.append([five_true[i],five_bp[i],five_lstm[i]])
            for i in range(len(five_bp)):
                recent = models.Prediction_com(data=data[i],true_date=round(five_true[i],2),bp_predict=round(five_bp[i],2),lstm_predict=round(five_lstm[i],2),bp_error=round(five_bp[i]-five_true[i],2),lstm_error=round(five_lstm[i]-five_true[i],2))
                five.append(recent)
    # fit and predict the value)
            lstm_predict=None
            try:
                model_lstm=model.lstm_fit(train,target)
                lstm_predict = model.lstm_predict(model_lstm, remain_predict)
            except:
                lstm_predict=fusion_predict
    #judge the accuracy
            if lstm_predict/bp_predict>1.08 or lstm_predict/bp_predict<0.92:
                lstm_predict = fusion_predict
            lstm_predict,bp_predict=lstm_predict.tolist()[0],bp_predict.tolist()[0]
        else:
#            tmp_price=model.get_price(stock_flag,stock_id)
#            lstm_predict,bp_predict=model.get_api_price(tmp_price)
            lstm_predict, bp_predict=0.0,0.0
        print lstm_benifit,fusion_benifit
        prices=models.Prices(lstm_price=lstm_predict,bp_price=bp_predict)
        if flag:
            company_name=model.get_commpany_name(param)
        else:
            company_name=param
        tmp_name=company_name.replace("A","").replace("B","").replace("*","")
        related_news=model.get_related_news(tmp_name)
        tendency=None
        if related_news:
            tendency=model.get_news_tendency(related_news)
        news=[]
        result = model.get_result(bp_predict, lstm_predict, last_price,tendency)
        if type(company_name)==list:
            company_name=company_name[0]
        if tendency:
            for i in range(len(tendency)):
                new=models.News(title=related_news[i][0],url=related_news[i][1],tentency=tendency[i][0])
                news.append(new)
        return render(request,'test1.html',{'prediction_prices':prices, 'news': news,'stock_id':stock_id,'company_name':company_name,'five':five,'last_price':last_price,'fusion_benifit':fusion_benifit,'lstm_benifit':lstm_benifit,'result':result})
    else:
        return render(request,'nothing.html')
