import json
import logging

import cv2
import os

import keras
import numpy as np
from djangoface.Face.net.mtcnn import mtcnn
from djangoface.Face.utils import utils as utils
from djangoface.Face.net.inception import InceptionResNetV1

keras.backend.clear_session()
def map_to_float(s):
    return float(s)


class face_rec():
    def __init__(self):
        # 创建mtcnn对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5, 0.8, 0.9]

        # 载入facenet
        # 将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        # model.summary()
        model_path = 'model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        # -----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        # -----------------------------------------------#
        face_list = os.listdir("face_dataset")

        self.known_face_encodings = []

        self.known_face_names = []

        for face in face_list:
            try:
                name = face.split(".")[0]
                img = cv2.imread("./face_dataset/" + face)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 检测人脸
                rectangles = self.mtcnn_model.detectFace(img, self.threshold)
                # 转化成正方形
                rectangles = utils.rect2square(np.array(rectangles))
                # facenet要传入一个160x160的图片
                rectangle = rectangles[0]
                # 记下他们的landmark
                landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                        rectangle[3] - rectangle[1]) * 160

                crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = cv2.resize(crop_img, (160, 160))
            except BaseException:
                print("提醒：处理图片：" + face + "时出错")
                continue
            new_img, _ = utils.Alignment_1(crop_img, landmark)

            new_img = np.expand_dims(new_img, 0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

    def calc_128_vec(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 检测人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)
            # 转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下他们的landmark
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)

            new_img = np.expand_dims(new_img, 0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)
        except BaseException as e:
            logging.exception(e)
            face_encoding = "-1"
        return face_encoding

    def calc_vec(self, img_path):
        code = 1
        res = None
        try:
            draw = utils.convert_img(img_path)
            res = self.calc_128_vec(draw)
            res = ",".join(str(it) for it in res.tolist())
            if res == "-1" or res is None:
                code = 0
        except Exception:
            code = 0
        return json.dumps({"code": code, "vec": res})

    def calc_distance(self, vecs, img_path):
        code = 1
        res = None
        vecs = vecs.split("~")
        try:
            draw = utils.convert_img(img_path)
            other_vec = self.calc_128_vec(draw)
            arr = []
            for vec in vecs:
                vec = vec.split(",")
                vec = list(map(map_to_float, vec))
                vec = np.array(vec)
                arr.append(vec)
            res = utils.face_distance(other_vec, arr)
            num = 0
            min = res[0]
            for t in res:
                if (t <= 0.8):
                    num += 1
                min = t if t < min else min
            # if min >= 0.6:
            #     code = 0
            if num < 3:
                code = 0
            print(res)
        except Exception as e:
            logging.exception(e)
            code = 0
        return json.dumps({"code": code, "distance": min})


