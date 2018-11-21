# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Prices(models.Model):
    bp_price=models.IntegerField(null=False)
    lstm_price=models.IntegerField(null=True)

class News(models.Model):
    title=models.TextField()
    tentency=models.TextField()
    url=models.TextField()

class Prediction_com(models.Model):
    true_date=models.FloatField()
    bp_predict=models.FloatField()
    lstm_predict = models.FloatField()
    bp_error=models.FloatField()
    lstm_error=models.FloatField()
    data=models.TextField()



