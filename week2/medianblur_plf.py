#!/user/bin/python
# -*- coding:utf-8 -*-
# @Author :Lief

import numpy as np
import matplotlib.pyplot as plt
import cv2


def padding(img_channel, kernel, padding_way):
	"""

	:param img_channel:the channel of the img,图片处理的通道矩阵值
	:param kernel:the size of kernel，卷积核维度
	:param padding_way:padding the img，经过卷积核处理后图片填的充方式
	"""
	height,width = img_channel.shape
	left = int((kernel-1)/2)
	# 对图片左侧和上侧的填充行数
	right = kernel-left-1
	# 对图片右侧和下侧的填充行数
	if padding_way == 0:
		left_padding = np.zeros((height,left))
		# 左侧填充0
		right_padding = np.zeros((height,right))
		# 右侧填充0
		img_channel = np.hstack((left_padding, img_channel, right_padding))
		# 先水平左右拼接上面矩阵
		uper_padding = np.zeros((left,width+left+right))
		# 上侧填充0
		down_padding = np.zeros((right,width+left+right))
		# 下侧填充0
		img_channel = np.vstack((uper_padding, img_channel, down_padding))
		# 再进行上下拼接填充矩阵
	elif padding_way == 1:
		left_padding = np.tile(img_channel[:, 0].reshape(height, 1), left)
		# 左侧填充：将img_channel的矩阵左侧列复制left次
		right_padding = np.tile(img_channel[:, -1].reshape(height, 1), right)
		# 右侧填充：将img_channel的矩阵右侧列复制right次
		img_channel = np.hstack((left_padding,img_channel,right_padding))
		# 先水平左右拼接上面矩阵
		uper_padding = np.tile(img_channel[0, :].reshape(1, width), left)
		# 上侧填充：将img_channel的矩阵上侧列复制left次
		down_padding = np.tile(img_channel[-1, :].reshape(1, width), right)
		# 下侧填充：将img_channel的矩阵下侧列复制right次
		img_channel = np.vstack((uper_padding,img_channel,down_padding))
	else:
		pass
	return img_channel


def MedianBlur(img,kernel,padding_way= 0):
	"""

	:param img:input img
	:param kernel:the size of the MedianBlur
	:param padding_way:chose of the padding way
	"""
	RGB = cv2.split(img)
	height, width, depth = img.shape
	new_img = np.zeros((3, height+kernel-1, width+kernel-1))
	# 图片经过kernel卷积处理后长宽会缩减kernel-1，为保证原图尺寸需先加上该值
	if padding_way == 0:
		for i, img_channel in enumerate(RGB):
			# 其中i是0,1,2中一个，img_channel是RGB中一个
			new_img[i] = padding(img_channel, kernel, 0)
	else:
		for i, img_channel in enumerate(RGB):
			new_img[i] = padding(img_channel, kernel, 1)
	blur_img = np.zeros((3, height, width))
	for i, img_channel in enumerate(new_img):
		for m in range(height):
			for n in range(width):
				blur_img[i][m][n] = np.median(img_channel[m:m+kernel-1, n:n+kernel-1])
				# 将blur_img的每个通道的像素和new_img每个通道的像素经过均值处理后相对应
	blur_img = cv2.merge(blur_img).astype("uint8")
	return blur_img


def my_show(img):
	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


if __name__ =="__main__":
	test_img = np.random.randint(0, 10, (4, 5))
	print(test_img)
	print(padding(test_img, 6, 0))
	img_noisy = cv2.imread("noisy_lenna.jpg",1)
	plt.figure(figsize=(6, 6))
	plt.subplot(121)
	my_show(img_noisy)
	plt.subplot(122)
	mb_lenna = MedianBlur(img_noisy, 7, 0)
	my_show(mb_lenna)
	plt.show()

"""
1.RANSAC算法简介
	当我们从估计模型参数时，用p表示一些迭代过程中从数据集内随机选取出的点均为局内点的概率；此时，结果模型很可能有用，
因此p也表征了算法产生有用结果的概率。用w表示每次从数据集中选取一个局内点的概率，如下式所示：
	w = 局内点的数目 / 数据集的数目
	通常情况下，我们事先并不知道w的值，但是可以给出一些鲁棒的值。假设估计模型需要选定n个点，wn是所有n个点均为局内点
的概率；1−w^n是n个点中至少有一个点为局外点的概率，此时表明我们从数据集中估计出了一个不好的模型。(1−w^n)k表示算法永远
都不会选择到n个点均为局内点的概率，它和1-p相同。因此，1−p=(1−w^n)k。我们对上式的两边取对数，得出:k=log(1−p)/log(1−w^n)
	值得注意的是，这个结果假设n个点都是独立选择的；也就是说，某个点被选定之后，它可能会被后续的迭代过程重复选定到。这种
方法通常都不合理，由此推导出的k值被看作是选取不重复点的上限。例如，要寻找适合的直线，RANSAC算法通常在每次迭代时选取2个点，
计算通过这两点的直线可能模型，要求这两点必须唯一。

2.RANSAC伪代码
input:
        data - a set of observations 一组观测数据
        model - a model that can be fitted to data 适用于数据的模型
        n - the minimum number of data required to fit the model 适用于模型的最少数据个数
        k - the number of iterations performed by the algorithm 算法的迭代次数
        t - a threshold value for determining when a datum fits a model 决定数据是否适用于模型的阈值
        d - the number of close data values required to assert that a model fits well to data 决定模型是否适用于数据集的数据数目

output:
        best_model - model parameters which best fit the data (or null if no good model is found)
        # 和数据最匹配的模型参数（如果没有找到好的模型，返回null）
        best_consensus_set - data point from which this model has been estimated 估计出模型的数据点
        best_error - the error of this model relative to the data 与数据相关的模型误差

    iterations = 0

    best_model = null

    best_consensus_set = null

    best_error = infinity

    while iterations < k
        maybe_inliers = n randomly selected values from data  从数据集中随机选择n个内点
        maybe_model = model parameters fitted to maybe_inliers 适合这n个内点的模型参数
        consensus_set = maybe_inliers

        for every point in data not in maybe_inliers  对于每一个不在maybe_inliers 中的数据点
           if point fits maybe_model with an error smaller than t 如果该点适合maybe_model，并且误差小于阈值
               add point to consensus_set  就将该点加入到consensus_set中

        if the number of elements in consensus_set is > d 
        # 如果consensus_set中的数据点个数大于d，这暗示着已经找到了好的模型，现在测试该模型到底多好
           (this implies that we may have found a good model,now test how good it is)
           better_model = model parameters fitted to all points in consensus_set 适合consensus_set中所有点的模型参数
           this_error = a measure of how well better_model fits these points 衡量适合这些点的better_model有多好
           if this_error < best_error 比最好的误差还小，说明发现了比之前好的模型，保存该模型直到有更好的模型
              (we have found a model which is better than any of the previous ones,keep it until a better one is found)
               best_model = this_model
               best_consensus_set = consensus_set
               best_error = this_error

        increment iterations迭代次数增加

    return best_model, best_consensus_set, best_error
"""