# 导入所需模块
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.initializers import TruncatedNormal
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K, utils, callbacks
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras import backend as K
K.set_image_dim_ordering("tf")
# 训练样本目录和测试样本目录
train_dir = './data/train/'
test_dir = './data/validation/'
# 对训练图像进行数据增强
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# 对测试图像进行数据增强
test_pic_gen = ImageDataGenerator(rescale=1./255)
# 利用 .flow_from_directory 函数生成训练数据
train_flow = train_pic_gen.flow_from_directory(train_dir,
                                               target_size=(24,24),
                                               batch_size=64,
                                               class_mode='categorical')
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(test_dir,
                                             target_size=(24,24),
                                             batch_size=64,
                                             class_mode='categorical')
tbCallBack = callbacks.TensorBoard(log_dir='./logs/1',
                                         histogram_freq= 0,
                                         write_graph=True,
                                         write_images=True)
def Model():
    model = Sequential()
    inputShape = (24, 24, 3)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (3, 24, 24)
        chanDim = 1

    # 第一层
    model.add(Conv2D(32, (3, 3),
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二层
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #第三层
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FC层
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    # output layer
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model
#检测
model=Model()
#model.summary()
"""
his = model.fit_generator(train_flow,
                    steps_per_epoch=20,
                    epochs=150,
                    verbose=1,
                    validation_data=test_flow,
                    validation_steps=20,
                    callbacks=[tbCallBack])
model.save('./weights/model.h5')
plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
