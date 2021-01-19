'''
通过这个文件进行网络截取的图片的预处理，更改文件路径上的文件名称即可处理该文件
'''
from skimage import io,transform,color

for i in range(10):
    img_name="E:\\GITHUB\\Q_learning\\test\\%s.jpg"%str(i)

    img=io.imread(img_name)#读入

    new_img=transform.resize(img,(128,128))#缩小成128*128

    img_gray = color.rgb2grey(new_img)#转灰度图

    io.imsave("E:\\GITHUB\\Q_learning\\test\\%s.jpg"%str(i),img_gray)#保存