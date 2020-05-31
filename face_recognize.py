import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1


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
        model_path = './model_data/facenet_keras.h5'
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
            name = face.split(".")[0]
            img = cv2.imread("./face_dataset/" + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)

            # 转化成正方形
            try:
                rectangles = utils.rect2square(np.array(rectangles))
            except IndexError:
                print("提醒：处理图片：" + face + "时出错")
                continue
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下他们的landmark
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            try:
                crop_img = cv2.resize(crop_img, (160, 160))
            except BaseException as ee:
                print("提醒：处理图片：" + face + "时出错")
                continue
            new_img, _ = utils.Alignment_1(crop_img, landmark)

            new_img = np.expand_dims(new_img, 0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

    def recognize(self, draw):
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

            # 输出检测图片中每张人脸的特征向量
            print("128维特征向量：", face_encoding)

            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)

            print("输出对应数据库所有图片的距离得分（最小值为最接近）：", face_distances)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                print("数据库中距离最近图片的特征向量：", self.known_face_encodings[best_match_index])
            arr = []
            arr.append(self.known_face_encodings[best_match_index])
            print("相似度：", utils.face_distance(arr, face_encoding))
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        # -----------------------------------------------#
        #   画框~!~
        # -----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left, bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw

    def use_cv():
        dududu = face_rec()
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, draw = video_capture.read()
            dududu.recognize(draw)
            cv2.imshow('Video', draw)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

    def calc_128_vec(self, img_path):
        img = cv2.imread(img_path)
        utils.calc_128_vec(self.facenet_model, img)

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

        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            print("输出对应数据库所有图片的距离得分（最小值为最接近）：", face_distances)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            arr = []
            arr.append(self.known_face_encodings[best_match_index])
            tmp = utils.face_distance(arr, face_encoding)
            if tmp < min:
                min = tmp
                res = face_encoding
        return res

    def clac_vec(self, img_path):
        draw = utils.convert_img(img_path)
        res = self.calc_128_vec(draw)
        return res

    def clac_distance(self, vec, img_path):
        matches = utils.compare_faces(self.known_face_encodings, vec, tolerance=0.9)
        # 找出距离最近的人脸
        face_distances = utils.face_distance(self.known_face_encodings, vec)
        # 取出这个最近人脸的评分
        best_match_index = np.argmin(face_distances)
        arr = []
        arr.append(self.known_face_encodings[best_match_index])
        res = utils.face_distance(arr, vec)
        return res[0]


if __name__ == '__main__':
    dududu = face_rec()
    vec = dududu.clac_vec("D:/faceR-master/face_dataset/wangbo.JPG")
    print(dududu.clac_distance(vec, ""))
