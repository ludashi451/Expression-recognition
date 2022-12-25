import cv2 as cv
import os
img_file = "D://python_project_2022//test0903//pictures"


def face_detection(image):

    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    cascade = cv.CascadeClassifier("D://python_project_2022//test0903//haarcascade_frontalface_default.xml")
    face_detecter = cascade
    # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
    faces = face_detecter.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=5)
    print('检测人脸信息如下：\n', faces)
    index = 0
    for x, y, w, h in faces:
        # 在原图像上绘制矩形标识
        cv.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
        cv.imshow('result', image)
        print(faces[index][0])
        print(faces[index][1])
        print(faces[index][2])
        print(faces[index][3])
        xx = faces[index][0]
        yy = faces[index][1]
        ww = faces[index][2]
        hh = faces[index][3]


        result = src[yy:hh+yy , xx:ww+xx ]
        # result = src[140:324, 324:439]

        cv.imshow("sss", result)
        # if not os.path.exists(img_file):
        #     os.mkdir(img_file)
        cv.imwrite("D://python_project_2022//test0903//pictures//result00.jpg", result)


        index = index+1



src = cv.imread(r'images/3.jpg')
cv.imshow('input image', src)



face_detection(src)
cv.waitKey(0)
cv.destroyAllWindows()