# -*- coding: utf-8 -*-
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators import csrf
 
# 接收POST请求数据
def search_post(request):
    ctx ={}
    name = request.body
    print(name)
    if request.POST:
        ctx['rlt'] = request.POST['q']
    print(ctx)
    return render(request, "post.html", ctx)