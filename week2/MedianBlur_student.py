#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_show(img):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def padding(color,kernel,flag=0):
    """
    :param color: r,g,b color，
    :param kernel: kernel size，此处是卷积核维度，不是卷积核
    :param flag: padding_way
    :return: new r,g,b
    """
    height,width=color.shape    # 图片长宽和通道数
    left = int((kernel - 1) / 2)
    # 左侧和上侧的填充行数
    right = kernel - 1 - left
    # 右侧和下侧的填充行数

    if flag ==0:    # 0样式填充
        left_padding = np.zeros((height, left))
        right_padding = np.zeros((height, right))
        # color = np.hstack((left_padding, color, right_padding))   # np.hstack是沿水平方向插入合并数组
        color = np.concatenate((left_padding, color, right_padding),axis=1)
        # 左右进行填充，此处np.concatenate方法和np.hstack方法类似
        up_padding = np.zeros((left,width+kernel-1,))
        down_padding = np.zeros((right,width+kernel-1))
        # color = np.vstack((up_padding, color, down_padding))
        color = np.concatenate((up_padding, color, down_padding),axis=0)
        # 上下进行填充，此处np.concatenate方法和np.vstack方法类似
    elif flag==1:   # 将color四边分别向外复制式填充
        left_padding = np.tile(color[:,0].reshape(height,1),left)
        # np.tile(A,rep)是将A按照rep的形式进行重复
        # 其中A是color[:,0].reshape(height,1)表示color第一列数据重新构成（height，1）形矩阵，rep重复次数为left
        right_padding = np.tile(color[:,-1].reshape(height,1),right)
        # color[:,-1].reshape(height,1)表示color最后一列数据重新构成（height，1）形矩阵，重复次数为right
        color = np.concatenate((left_padding, color, right_padding), axis=1)
        up_padding = np.tile(color[0,:],(left,1))
        # np.tile(color[0,:],(left,1))将color的第一行进行（left样式重复）
        down_padding = np.tile(color[-1,:],(right,1))
        # np.tile(color[-1,:],(left,1))将color的最后一行进行（left样式重复）
        color = np.concatenate((up_padding, color, down_padding), axis=0)
    else:
        pass

    return color

def medianBlur(img, kernel, padding_way='ZEROS'):
    """img & kernel is List of List; padding_way a string
    when "REPLICA" the padded pixels are same with the border pixels
    when "ZERO" the padded pixels are zeros"""
    height,width,depth=img.shape
    # shape是图片高度x宽度x通道
    rgb = cv2.split(img)
    # 分解成rgb后对每个通道进行处理
    # 如果不填充，步幅为1的情况下卷积后的图像宽为width-kernel+1，因此填充的宽度为kernel-1
    new_img = np.zeros((3,height+kernel-1,width+kernel-1))
    # 构造长宽分别为height+kernel-1与width+kernel-1像素为0的图片3通道矩阵
    if padding_way == 'ZEROS':
        for i,c in enumerate(rgb):
        # enumera（）是将rgb中元素一次遍历，角标复制给i，值赋给c
        # 此处的i是0,1,2,c是rgb中三通道中一个
            new_img[i] = padding(c,kernel,0)
            # new_img获得img填充的数组矩阵
    elif padding_way == 'REPLICA':
        for i, c in enumerate(rgb):
            new_img[i] = padding(c,kernel,1)
    else:
        pass

    blur_img = np.zeros((3,height,width))
    # 设置模糊处理的矩阵，初始值赋0
    for t,c in enumerate(new_img):
        # 对于每个通道的高和宽
        for i in range(height):
            for j in range(width):
                blur_img[t][i][j]= np.median(c[i:i+kernel,j:j+kernel])
                # 将new_img进行中值模糊化然后复制给blur_img，注意new_img长宽维度比blur_img维度大kernel-1
    blur_img=cv2.merge(blur_img).astype('uint8')
    # 将拆分获得矩阵合并后获得blur_img图像
    return blur_img


if __name__ == '__main__':
    # 该语句表明当上面三个函数模块直接调用的时候下面几段语句将会被执行，当上面的模块是被导入的时候下面的语句将不被执行。
    # 其实就是为了将下面语句选择性执行，在被其他程序调用上面模块时就不执行下面语句。
    img = np.random.randint(0,10,(4,5))
    print(padding(img,7,0))
    img = cv2.imread('noisy_lenna.jpg')
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    my_show(img)
    plt.subplot(122)
    newimg = medianBlur(img,7,padding_way='ZEROS')
    my_show(newimg)
    plt.show()
