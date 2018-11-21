# stock_prediction
# 基于LSTM的股票价格预测
## 目录说明
stock_prediction为项目根目录，项目使用django框架做了web界面，code.py是配置文件，有效代码在stock_prediction/django/myblog/blog中，其中模块tendency_judge用来对股票相关新闻进行分类
## 实验原理
使用LSTM和BP神经网络进行股票价格的回归，时间窗口设置为120，根据前120天的数据，预测一个交易日的股票价格，根据股票相关新闻的分类结果对模型预测价格进行奖惩，得出最终的股票预测价格。
## 启动方式
在目录stock_prediction/django/myblog/下执行 python manage.py runserver ,复制页面的url至浏览器即可访问系统主页 
