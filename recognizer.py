import cv2
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn import mtcnn
from predict import Model
import numpy as np
from matplotlib.pyplot import MultipleLocator
import datetime
from skimage import exposure
from emomodel import build
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model=mtcnn()
threshold = [0.5, 0.6, 0.7]
eyes=[]
num_frames=[]
mouth=[]
cap = cv2.VideoCapture(0)
emomodel=build(width=128, height=128, depth=3,classes=3)
emomodel.load_weights('face_model.h5')

#绘制眼睛，嘴巴
def plot_eyes(eys,num_frames,mouth,name):
    i = datetime.datetime.now()
    y_major_locator = MultipleLocator(1)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.scatter(num_frames, eyes,c='b')
    for num in range(len(mouth)):
        if mouth[num]>0:
            plt.scatter(num,mouth[num],c='r')
    plt.title('moniter on the screens')
    plt.axis('off')
    plt.savefig("data/"+name+"/"+name+"_"+str(i.month)+str(i.day)+'.png')
    plt.show()

def tell_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma_img = exposure.adjust_gamma(gray, 0.7)
    ret, binary = cv2.threshold(gamma_img, 127, 255, cv2.THRESH_BINARY)
   # cv2.imwrite('eyes/eyes_%d.jpg'%i,binary)
   # cv2.imwrite('eyes/eyes_%d.jpg' % i,gray)
    return(binary[25,25]+binary[26,25]+binary[27,25]+binary[25,26]+binary[25,27])
def tell_mouth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 102, 255, cv2.THRESH_BINARY)
    return len(binary[binary==0])/len(binary)

def tell_emo(image):
    test_image = cv2.resize(image, (128, 128))
    test_image = np.asarray(test_image.astype("float32"))
    test_image = test_image / 255.
    test_image = test_image.reshape((1, 128, 128, 3))
    preds = emomodel.predict_classes(test_image)
    return preds





def main_rec(name):
    score = True
    pred = 0
    i=0
    k=0
    num=0
    while num<50:
        ret, frame = cap.read()
        rectangles = model.detectFace(frame, threshold)
        print(rectangles)
        num+=1
        if len(rectangles)>0:
            for rectangle in rectangles:
                if rectangle is not None:
                    W = -int(rectangle[0]) + int(rectangle[2])
                    H = -int(rectangle[1]) + int(rectangle[3])
                    paddingH = 0.01 * W
                    paddingW = 0.02 * H
                    crop_img = frame[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                               int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                    if crop_img is None:
                        continue
                    if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                        continue
                    # cv2.rectangle(frame, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                    # (255, 0, 0), 1)
                    #表情判断
                    face_image=crop_img
                    emo_value=tell_emo(face_image)
                    if emo_value == [0]:
                        cv2.putText(frame, 'distracted', (20, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)
                    elif emo_value==[1]:
                        cv2.putText(frame, 'interested', (20, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (155, 255, 255), 2)
                    else:
                        cv2.putText(frame, 'normal', (20, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 155, 255), 2)
                    #眼睛判断
                    lefteyes = (int(rectangle[5]), int(rectangle[6]))
                    cv2.rectangle(frame, (int(rectangle[5] - 25), int(rectangle[6] + 25)),
                                  (int(rectangle[5] + 25), int(rectangle[6] - 25)), (255, 255, 0), 1)
                    eyes_image = frame[int(rectangle[6] - 25):int(rectangle[6] + 25),
                                 int(rectangle[5] - 25):int(rectangle[5] + 25)]
                    # cv2.imwrite("eyes_%d.jpg"%num,eyes_image)
                    # print(eyes_image.shape)
                    eyes_value = tell_eyes(eyes_image)

                    if eyes_value == 0:
                        cv2.putText(frame, 'open_eyes', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
                        i = 1
                        eyes.append(i)
                    else:
                        cv2.putText(frame, 'closed_eyes', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)
                        i = 0
                        eyes.append(i)
                    cv2.circle(frame, lefteyes, 1, (0, 0, 255), 1)
                    righteyes = (int(rectangle[7]), int(rectangle[8]))
                    cv2.circle(frame, righteyes, 1, (0, 0, 255), 1)
                    nose = (int(rectangle[9]), int(rectangle[10]))
                    cv2.circle(frame, nose, 1, (0, 0, 255), 1)
                    leftmouth = (int(rectangle[11]), int(rectangle[12]))
                    cv2.circle(frame, leftmouth, 1, (0, 0, 255), 1)
                    rightmouth = (int(rectangle[13]), int(rectangle[14]))
                    cv2.circle(frame, rightmouth, 1, (0, 0, 255), 1)
                    #嘴巴判断
                    mouth_img = frame[int(rectangle[14] - 30):int(rectangle[14] + 30),
                                int(rectangle[11]):int(rectangle[13])]
                    mouth_value=tell_mouth(mouth_img)
                    if mouth_value>25:
                        cv2.putText(frame, 'yawn', (250, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 2)
                        k=0.5
                        mouth.append(k)
                    else:
                        k=0
                        mouth.append(k)
        else:
            i=0
            k=0
            eyes.append(i)
            mouth.append(k)
        # default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for num in range(len(eyes)):
        num_frames.append(num)
    plot_eyes(eyes, num_frames, mouth,name)
   # print('eyes',eyes)
    #print('num_frames',num_frames)
    #print(mouth)
    cap.release()
    cv2.destroyAllWindows()




main_rec('lilei')