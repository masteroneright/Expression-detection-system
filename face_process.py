
"""
model=mtcnn()
threshold = [0.5, 0.6, 0.7]
def face_point(img):
    rectangles = model.detectFace(img, threshold)#检测人脸
    if len(rectangles)>0:
            for rectangle in rectangles:
                if rectangle is not None:
                    W = -int(rectangle[0]) + int(rectangle[2])
                    H = -int(rectangle[1]) + int(rectangle[3])
                    paddingH = 0.01 * W
                    paddingW = 0.02 * H
                    face_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                               int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]#人脸区域
                    eyes_img = img[int(rectangle[6] -50):int(rectangle[6] + 50),
                                 int(rectangle[5] - 40):int(rectangle[5] + 40)]#人眼+眉毛区域
                    mouth_img = img[int(rectangle[14] - 30):int(rectangle[14] + 30),
                                int(rectangle[11]):int(rectangle[13])]
                    #meimao_point=(rectangle[5]+rectangle[7])/2
                    #zhoumei_img=img[int(rectangle[6] -50):int(rectangle[6] + 70), int((meimao_point - 40)):int((meimao_point + 60))]
    else:
        face_img, eyes_img, mouth_img=img,img,img


    return  face_img,eyes_img,mouth_img
#test

print(img.shape)
face_img,eyes_img,mouth_img,zhoumei_img=face_point(img)
gray = cv2.cvtColor(eyes_img, cv2.COLOR_BGR2GRAY)
gamma_img = exposure.adjust_gamma(gray, 0.5)
ret, binary = cv2.threshold(gamma_img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
"""
"""
eyes_img=cv2.imread('eyes.jpg')
gray = cv2.cvtColor(eyes_img, cv2.COLOR_BGR2GRAY)
gamma_img = exposure.adjust_gamma(gray, 0.5)
ret, binary = cv2.threshold(gamma_img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(eyes_img, contours, -1, (0, 0, 255), 3)
print(len(contours))
cv2.imshow('lunkuo',eyes_img)
cv2.imshow('binary',binary)
#cv2.imshow('face',face_img)
#cv2.imshow('eyes',eyes_img)
#cv2.imwrite('eyes.jpg',eyes_img)
#cv2.imshow('mouth',mouth_img)
#cv2.imshow('zoumei',zhoumei_img)
#print(contours)
"""
import cv2
import dlib
import numpy as np
#dlib 人脸识别模型
#facerec=dlib.face_recognition_model_v1("model")
#Dlib 探测器
detector=dlib.get_frontal_face_detector()
#68点特征点模型
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
"""
img=cv2.imread('face2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度
faces=detector(gray,1)
print(faces)
print("发现｛0｝个人脸".format(len(faces)))
points=[]
for (index,face) in enumerate(faces):
    shape=predictor(img,face)#68个特征点

    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
       # print(pt_pos)
        points.append(pt_pos)
        cv2.rectangle(img,(face.left(),face.top()),(face.right(),face.bottom()),(0,255,0),2)
eyes_img = img[int(points[21][1]):int(points[0][1]),int(points[17][0]):int(points[27][0])]  # 人眼+眉毛区域
mouth_img = img[int(points[33][1]):int(points[11][1]),int(points[48][0]):int(points[54][0])]
eyes_to_brow=points[37][1]-points[19][1]
left_to_right=points[45][0]-points[37][0]
#print(len(points))
print(left_to_right)
cv2.imshow('aa',mouth_img)
cv2.waitKey(0)
#cv2.waitKey(0)
"""
#Dlib 探测器
#detector=dlib.get_frontal_face_detector()
#68点特征点模型
#predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def face_point(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度
    points=[]
    faces = detector(gray, 1)
    for (index, face) in enumerate(faces):
        shape = predictor(img, face)  # 68个特征点
        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)#关键点坐标
            points.append(pt_pos)
    face_img = img[int(face.top()):int(face.bottom()),
               int(face.left()):int(face.right())]
    eyes_img = img[int(points[21][1]):int(points[2][1]), int(points[17][0]):int(points[27][0])]
    mouth_img = img[int(points[33][1]):int(points[11][1]), int(points[48][0]):int(points[54][0])]
    eyes_to_brow = points[37][1] - points[19][1]
    left_to_right = points[45][0] - points[37][0]
    ratio=float(eyes_to_brow/left_to_right)
    return face_img,eyes_img,mouth_img,ratio
#test
img=cv2.imread('face.jpg')
face_img,eyes_img,mouth_img,ratio=face_point(img)
cv2.imshow('face',face_img)
cv2.imshow('eyes',eyes_img)
print(ratio)
cv2.waitKey(0)





