"""
visualize results for test image
"""
# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import cv2 as cv
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

result = cv.imread("pictures/result.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



# 打开摄像头
cap = cv.VideoCapture(0)
# result = np.zeros((48, 48))
while (True):

    # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
    hx, frame = cap.read()

    # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
    if hx is False:
        # 打印报错
        print('read video error')
        # 退出程序
        exit(0)
    # 显示摄像头图像，其中的video为窗口名称，frame为图像
    # cv.imshow('video', frame)


    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    cascade = cv.CascadeClassifier("D://python_project_2022//test0903//haarcascade_frontalface_default.xml")
    face_detecter = cascade
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    faces = face_detecter.detectMultiScale(image=frame, scaleFactor=1.1, minNeighbors=5)
    # print('检测人脸信息如下：\n', faces)
    index = 0

    for x, y, w, h in faces:
        # 在原图像上绘制矩形标识
        cv.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)

        # print(faces[index][0])
        # print(faces[index][1])
        # print(faces[index][2])
        # print(faces[index][3])
        xx = faces[index][0]
        yy = faces[index][1]
        ww = faces[index][2]
        hh = faces[index][3]
        # global result

        result = frame[yy:hh + yy, xx:ww + xx]
    # result = src[140:324, 324:439]



    # result = detection_face(faces)
    gray = rgb2gray(result)


    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
    # 加一层维度
    img = gray[:, :, np.newaxis]

    # 进行合并,axis等于2就是对第二维度进行合并，可参考李沐大神的理解
    img = np.concatenate((img, img, img), axis=2)

    # 转换成Image格式
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG19')
    # checkpoint = torch.load(os.path.join('CK+_VGG19', '1', 'Test_model.t7'))
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model22927_70.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    with torch.no_grad():
        inputs = Variable(inputs)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    # print(score.index(max(score)))  # 返回第一个最大值的位置

    _, predicted = torch.max(outputs_avg.data, 0)
    #
    # plt.rcParams['figure.figsize'] = (13.5, 5.5)
    # axes = plt.subplot(1, 3, 1)
    # plt.imshow(raw_img)
    # plt.xlabel('Input Image', fontsize=16)
    # axes.set_xticks([])
    # axes.set_yticks([])
    # plt.tight_layout()
    index22 = int(predicted.cpu().numpy())
    print(index22)

    if index22 == 4:
        cv.putText(frame, "sad", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("sad")
    if index22 == 3:
        cv.putText(frame, "happy", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("happy")
    if index22 == 0:
        cv.putText(frame, "angry", (100, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        print("angry")
    if index22 == 2:
        cv.putText(frame, "fear", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("fear")
    if index22 == 1:
        cv.putText(frame, "disgust", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("disgust")
    if index22 == 5:
        cv.putText(frame, "surprise", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("Surprise")
    if index22 == 6:
        cv.putText(frame, "Neutral", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("Neutral")
    if index22 == 100:
        cv.putText(frame, "出现问题", (100, 100), cv.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        print("出现问题")

    index22 = 100
    cv.imshow('video', frame)
    # 监测键盘输入是否为q，为q则退出程序

    if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        break

# 释放摄像头
cap.release()

# 结束所有窗口
cv.destroyAllWindows()





















