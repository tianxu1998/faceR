"""Face URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
# from django.urls import path

# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]
# from django.conf.urls import url
 
# from . import views
 
# urlpatterns = [
#     url(r'^$', views.hello),
# ]

# from django.urls import path
# from . import views
# urlpatterns = [
#     path('r/', views.runoob),
# ]
# from apscheduler.scheduler import Scheduler  
from django.conf.urls import url
from . import search,search2,stark,views
# from django.urls import path
from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules
stark.www()
urlpatterns = [
    url(r'^search-form$', search.search_form),
    url(r'^search$', search.search),
    url(r'^search-post$', search2.search_post),
    url(r'^index$', views.index),
    url(r'^upload$', views.upload),
    url(r'^uploadwithparam$', views.uploadwithparam),
]
# sched = Scheduler()  #实例化，固定格式
 
# # @sched.interval_schedule(seconds=3)  #装饰器，seconds=60意思为该函数为1分钟运行一次
  
# def mytask():  
#     print('guosong')  
  
# sched.start()  #启动该脚本
