# -*- coding: utf-8 -*-
"""
Created on 2021.01.19
将自己从网络上截取的图片混合初始数据集进行验证。
输出结果：
仕.png的预测结果是：仕
兵.png的预测结果是：兵
将.jpg的预测结果是：卒
象.jpg的预测结果是：卒
象.png的预测结果是：帅
车.jpg的预测结果是：马
车.png的预测结果是：卒

截取的图片的无一预测正确，足以见得此模型的泛化性非常之差。
@author: 沧海云帆
"""
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn
import keras
from keras.models import load_model
import os
import warnings
from keras.models import load_model
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

#结果输出字典
dic = {1:'帅',2:'仕',3:'相',4:'马',5:'炮',6:'车',7:'兵',8:'卒',9:'将',10:'象'}


#*********************************************************************************************************************************************#
scaler = StandardScaler()
path = "./test"
dirs = os.listdir( path )
im2array = []
for file in dirs:
    image_dir=os.path.join(path,file)
    pic = np.array(Image.open(image_dir))
    im = scaler.fit_transform(pic)#标准化
    im2array.append(im)
im_array = np.array(im2array).reshape(7,128,128,1)
#*********************************************************************************************************************************************#

model = load_model('model.h5') #加载模型
result = np.argmax(model.predict(im_array),axis=1)#预测

#输出结果
for i,j in zip(result,dirs):
    print(j+'的预测结果是：'+dic[i])

#*********************************************************************************************************************************************#