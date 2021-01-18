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
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
import os
import warnings
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

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
#print(Y[1:20])

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

#标签转换成独热编码
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
#print(X_train.shape)
#print(y_train.shape)

#batch_size=50    #每次喂给网络50组数据
#*********************************************************************************************************************************************#
#卷积神经网络模型，采用keras进行搭建
model = Sequential() #这里使用序贯模型，比较容易理解,序贯模型就像搭积木一样，将神经网络一层一层往上搭上去
#搭一个卷积层
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer1_con1',input_shape=(128, 128, 1)))
#再搭一层卷积层
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer1_con2'))
#搭一个最大池化层
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer1_pool'))
#dropout层可以防止过拟合，每次有25%的数据将被抛弃
model.add(Dropout(0.25))
#和上面的网络结构类似
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer2_con1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',name='layer2_con2'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer2_pool'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))  #该全连接层共128个神经元
model.add(Dense(10, activation='softmax')) #一共分为10类，所以最后一层有10个神经元，并且采用softmax输出
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])#定义损失值、优化器
#*********************************************************************************************************************************************#
#保存训练参数
Checkpoint=ModelCheckpoint(filepath='./cnn_model',monitor='val_acc',mode='auto' ,save_best_only=True)
#回调函数
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,mode='auto',  verbose = 1, factor=0.5, min_lr = 0.00001)
#TensorBoard可视化
TensorBoard=TensorBoard(log_dir='./log', write_images=1, histogram_freq=1)

#训练
model.fit(X_train, y_train,epochs=3,callbacks=[learning_rate_reduction,TensorBoard,Checkpoint])

#计算得分
[loss,accuracy] = model.evaluate(X_test, y_test)
print('\nTest Loss: ', loss)
print('\nTest Accuracy: ', accuracy)
#保存模型
#model.save("model_new.h5")
#*********************************************************************************************************************************************#