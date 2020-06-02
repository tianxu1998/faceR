import json

import cv2
import os
import numpy as np
from djangoface.Face.net.mtcnn import mtcnn
from djangoface.Face.utils import utils as utils
from djangoface.Face.net.inception import InceptionResNetV1


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

    def calc_128_vec(self, draw):
        res = 0
        min = 1 << 30
        # -----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        # -----------------------------------------------#
        height, width, _ = np.shape(draw)

        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles) == 0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)
        # -----------------------------------------------#
        #   对检测到的人脸进行编码
        # -----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)
            new_img = np.expand_dims(new_img, 0)

            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)

            face_encodings.append(face_encoding)
        if len(face_encodings) > 1:
            res = "-1"
        else:
            res = face_encodings[0]
        # for face_encoding in face_encodings:
        #     # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
        #     matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
        #     name = "Unknown"
        #     # 找出距离最近的人脸
        #     face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
        #     print("输出对应数据库所有图片的距离得分（最小值为最接近）：", face_distances)
        #     # 取出这个最近人脸的评分
        #     best_match_index = np.argmin(face_distances)
        #     arr = []
        #     arr.append(self.known_face_encodings[best_match_index])
        #     tmp = utils.face_distance(arr, face_encoding)
        #     if tmp < min:
        #         min = tmp
        #         res = face_encoding
        return res

    def calc_vec(self, img_path):
        code = 1
        res = None
        try:
            draw = utils.convert_img(img_path)
            res = self.calc_128_vec(draw)
            if res == "-1":
                code = 0
        except Exception:
            code = 0
        return json.dumps({"code": code, "vec": ",".join(str(it) for it in res.tolist())})

    def calc_distance(self, vec, img_path):
        code = 1
        res = None
        try:
            vec = vec.split(",")
            vec = list(map(map_to_float, vec))
            vec = np.array(vec)
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, vec)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            arr = []
            arr.append(self.known_face_encodings[best_match_index])
            res = utils.face_distance(arr, vec)
        except Exception:
            code = 0
        return json.dumps({"code": code, "distance": res[0]})
