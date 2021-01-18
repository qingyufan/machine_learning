#初代版本，可忽略，仅作纪念意义
'''
import numpy as np
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
# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

X_train = np.random.random((5000, 128, 128, 1))
print(X_train.shape)
#这里的数据模仿的是图片数据
y_train = keras.utils.to_categorical(np.random.randint(10, size=(5000, 1)), num_classes=10)
#这些照片总共有10个类别
print(y_train.shape)
X_test = np.random.random((100, 128, 128, 1))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

batch_size=28    #每次喂给网络28组数据
epochs=2

model = Sequential() #这里使用序贯模型，比较容易理解,序贯模型就像搭积木一样，将神经网络一层一层往上搭上去
#搭一个卷积层
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer1_con1',input_shape=(128, 128, 1)))
#再搭一层卷积层效果更好
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer1_con2'))
#搭一个最大池化层
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer1_pool'))
#dropout层可以防止过拟合，每次有25%的数据将被抛弃
model.add(Dropout(0.25))
#和上面的网络结构类似
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer2_con1'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer2_con2'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding = 'same', data_format='channels_last',name = 'layer2_pool'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  #该全连接层共128个神经元
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) 
#一共分为10类，所以最后一层有10个神经元，并且采用softmax输出

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
#定义损失值、优化器

model.fit(X_train, y_train)

[loss,accuracy] = model.evaluate(X_test, y_test)
print('\nTest Loss: ', loss)
print('\nTest Accuracy: ', accuracy)'''