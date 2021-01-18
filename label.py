# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:42:41 2020

@author: 沧海云帆
"""
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import warnings
from keras.models import load_model
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

'''
#预处理数据集，并标准化
#session = tf.compat.v1.Session()
#pic = np.array(Image.open('./象棋数据集/1-帅/1.png'))
#im = scaler.fit_transform(pic)
#print(im)
'''


#*********************************************************************************************************************************************#
scaler = StandardScaler()
path = "./象棋数据集"
dirs = os.listdir( path )
im2array = []
for file in dirs:
    pic_dir=os.path.join(path,file)  #子文件夹的路径
    for i in os.listdir(pic_dir):
        image_dir=os.path.join(pic_dir,i)  #子文件夹中图片的路径
        pic = np.array(Image.open(image_dir))
        im = scaler.fit_transform(pic)#标准化
        im2array.append(im)
im_array = np.array(im2array).reshape(7190,128,128,1)
#print(im_array.shape)
#*********************************************************************************************************************************************#
#生成标签
labels = np.zeros(shape=(7190,1),dtype=int)
for i in range(10):
    for j in range(719):
        labels[719*i+j][0] = i
#*********************************************************************************************************************************************#
#打乱数据集及数据标签
X,Y = sklearn.utils.shuffle(im_array,labels)
y = keras.utils.to_categorical(Y, num_classes=10)
#print(Y[1:20])
'''
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

#标签转换成独热编码
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
print(X_train.shape)
print(y_train.shape)
'''
#batch_size=50    #每次喂给网络50组数据
model = load_model('model.h5') #加载模型

#计算得分
[loss,accuracy] = model.evaluate(X, y)
print('\nTest Loss: ', loss)
print('\nTest Accuracy: ', accuracy)
#保存模型
#*********************************************************************************************************************************************#