from django.conf.urls import url,include
from . import views
urlpatterns = [
    url(r'index', views.index),
    url(r'', views.get_stock_id,name='get_stock_id')
]
