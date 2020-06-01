# from django.http import HttpResponse

# def hello(request):
#     return HttpResponse("Hello world ! ")

# from django.shortcuts import render

# def runoob(request):
#     context          = {}
#     context['hello'] = 'Hello World!'
#     return render(request, 'runoob.html', context)
# print('还是是说说')
# from django.shortcuts import render

# def runoob(request):
#     views_str = "<a href='https://www.baidu.com/'>点击跳转</a>"
#     return render(request, "runoob.html", {"views_str": views_str})
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
import os

from django.views.decorators.csrf import csrf_protect

from djangoface.Face import face_recognize as model


face = model.face_rec()


# def upload(request):
#     if request.method == "POST": # 请求方法为POST时，进行处理
#         myFile = request.FILES.get("myfile", None) # 获取上传的文件，如果没有文件，则默认为None
#         if not myFile:
#             return HttpResponse("no files for upload!")
#         destination = open(os.path.join(settings.MEDIA_ROOT, myFile.name),'wb+') # 打开特定的文件进行二进制的写操作
#         for chunk in myFile.chunks(): # 分块写入文件
#             destination.write(chunk)
#         destination.close()
#         return HttpResponse("upload over!")
def index(request):
    # if request.method == "POST":
    #     myFile = request.FILES.get("myfile")
    # ctx ={}
    # name = request.body
    # print(name)
    # if request.POST:
    #     ctx['rlt'] = request.POST['q']
    # print(ctx)
    return render(request, "index.html")

def upload(request):
    try:
        pic = request.FILES['file']
        path = os.path.join(settings.MEDIA_ROOT, pic.name)
        save = open(os.path.join(settings.MEDIA_ROOT, pic.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in pic.chunks():  # 分块写入文件
            save.write(chunk)
        # print(type(pic))
        save.close()
    except:
        return HttpResponse("no files for upload!")
    res = 0
    try:
        res = face.clac_vec(path)
    except Exception as e:
        res = "calc vec error"
        print(e)
    return HttpResponse(res)

def uploadwithparam(request):
    try:
        param = request.POST['param']
        pic = request.FILES['file']
        path = os.path.join(settings.MEDIA_ROOT, pic.name)
        save = open(os.path.join(settings.MEDIA_ROOT, pic.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in pic.chunks():  # 分块写入文件
            save.write(chunk)
        # print(type(pic))z
        save.close()
    except:
        return HttpResponse("no files for upload!")
    res = 0
    try:
        res = face.clac_distance(param, path)
    except Exception as e:
        print(e)
        res = "计算相似度错误"
    return HttpResponse(res)
