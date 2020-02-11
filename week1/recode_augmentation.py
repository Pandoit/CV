import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 1.读取操作
img_ori = cv2.imread('lenna.jpg', 1)  # 彩色图片
img_gray = cv2.imread('lenna.jpg', 0)  # 黑白图片
print(img_ori.shape)    # 图片长宽和通道数

# 1.1用CV2读取和打印操作，默认通道为 BGR
# img = cv2.imread("lenna.jpg", 1)  # 读取图片并转给img变量
# cv2.imshow("img", img_cv)  # 打印img变量并以img_cv名称显示图片
# key = cv2.waitKey()   # 27为esc键
# if key == 27:
#     cv2.destroyAllWindows()

# 1.2matplotlib 打印操作，默认通道为 RGB
plt.imshow(img_ori)
plt.figure(figsize=(8, 8))  # 规定图片大小为8*8英寸
plt.show()
print(img_ori)  # 打印出图像的三通道数值
print(img_ori.shape)  # 打印出图像的长宽尺寸h, w
print(img_ori.dtype)  # 显示图像中通道存储的格式：unit8

plt.imshow(img_gray)
plt.show()

plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))  # 将通道转换为 matplotlib 的默认格式 RGB，将BGR转换到RGB
plt.show()

# 1.3对比通道转换前后的效果图
plt.subplot(121)  # 打印图像为一行两列，此为第一列
plt.imshow(img_ori)
plt.subplot(122)  # # 打印图像为一行两列，此为第二列
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
plt.show()  # 只需要一个plt.show()就可以打印上面的两张图


# 1.4编写一个自己使用的 show 函数，自动进行通道转换和大小调整
def my_show(img, size=(4, 4)):  # 规定图片大小为4*4英寸
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# 2.图像裁剪 image crop
img_crop = img_ori[0:500, 100:500]  # 截取img_ori图片横坐标0-500，纵坐标100-500的部分
my_show(img_crop)   # 此处的my_show是函数，用到上面的函数封装

# 3.打印黑白图片
plt.figure(figsize=(2, 2))
plt.imshow(img_gray, cmap='gray')
plt.show()

# 4.对不同通道进行操作
B, G, R = cv2.split(img_ori)    # split函数分割img_ori图像通道，分割后的B,G,R都为矩阵数值
# cv2.imshow('B', B)
# cv2.imshow('G', G)
# cv2.imshow('R', R)
# img_combine = cv2.merge((B, G, R))    # cv2.merge函数合并图像通道
# cv2.imshow("combine", cv2.cvtColor(img_combine, cv2. COLOR_BGR2RGB))
# key = cv2.waitKey(0)  # 0为参数，单位为毫秒，便是间隔时间
# if cv2.waitKey(1) & 0xFF == ord('q'): # ord(' ')将字符转化为对应的整数（ASCII码），0xFF是十六进制常数
#     cv2.destroyAllWindows()

plt.figure(figsize=(5, 5))
plt.subplot(131)
plt.imshow(B, cmap='gray')
plt.subplot(132)
plt.imshow(G, cmap='gray')
plt.subplot(133)
plt.imshow(R, cmap='gray')
plt.show()

# 5.图像的存储形式
print(img_ori)  # 打印出图像的三通道数值
print(img_ori.dtype)  # 打印图片的数据类型，这里是 uint8
print(img_ori.shape)  # 打印图片的长宽


# 6.图像的灰阶平移color shift
def img_cooler(img, b_increase, r_decrease):    # b通道增，r通道减
    B, G, R = cv2.split(img)
    b_lim = 255 - b_increase    # b通道增加后的最大值
    B[B > b_lim] = 255  # numpy的语句操作，先判断再赋值
    B[B <= b_lim] = (b_increase + B[B <= b_lim]).astype(img.dtype)
    r_lim = r_decrease  # r通道减少后的最小值
    R[R < r_lim] = 0
    R[R >= r_lim] = (R[R >= r_lim] - r_decrease).astype(img.dtype)
    return cv2.merge((B, G, R))


img_cool = img_cooler(img_ori, 50, 0)
my_show(img_cool)   # 此处的my_show是函数,用于显示图片，用到上面1.4的函数封装

img_dark = cv2.imread('dark.jpg')
my_show(img_dark, size=(6, 6))


# 7.伽马校正 gamma correction:用于摄像机中对亮暗的调节
def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = []
	for i in range(256):
		table.append(((i / 255.0) ** invGamma) * 255)
	table = np.array(table).astype("uint8")
	return cv2.LUT(img_dark, table) # 将table中的值通过look up table映射到img_dark中


img_brighter = adjust_gamma(img_dark, 1.5)  # 设置值大于 1，整体亮度变亮
my_show(img_brighter, size=(6, 6))

# img_brighter = adjust_gamma(img_dark, 0.5)   # 设置值小于 1，整体亮度变暗
# my_show(img_brighter, size=(6, 6))

# 8.直方图操作 histogram
plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0, 256], color='b')  # img_dark.flatten()是将img_dark二维矩阵拉平成一维的
plt.subplot(122)
plt.hist(img_brighter.flatten(), 256, [0, 256], color='r')  # 256是指有256个条状数目，[0,256]是横坐标范围,纵坐标是各值出现次数
plt.show()

# 8.1YUV色彩空间的Y进行直方图均衡来调亮图片,YUV中Y是亮度参数
img_yuv = cv2.cvtColor(img_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # 只取Y通道并进行直方图均衡，0指的是第一个通道
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # y: luminance(明亮度), u&v: 色度饱和度

my_show(img_output, size=(6, 6))

plt.subplot(131)
plt.hist(img_dark.flatten(), 256, [0, 256], color='b')
plt.subplot(132)
plt.hist(img_brighter.flatten(), 256, [0, 256], color='r')
plt.subplot(133)
plt.hist(img_output.flatten(), 256, [0, 256], color='g')
plt.show()

# 9.相似变换 rotation
M = cv2.getRotationMatrix2D((img_ori.shape[1] / 2, img_ori.shape[0] / 2), 66, 0.8)  # center, angle, scale
img_rotate = cv2.warpAffine(img_ori, M, (img_ori.shape[1], img_ori.shape[0]))
my_show(img_rotate)

# 10.仿射变换 Affine Transform
rows, cols, ch = img_ori.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.7, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img_ori, M, (cols, rows))
my_show(dst)

# 11.投影变换 perspective transform
import random


def random_warp(img, row, col):
	height, width, channels = img.shape

	# warp:
	random_margin = 60
	x1 = random.randint(-random_margin, random_margin)
	y1 = random.randint(-random_margin, random_margin)
	x2 = random.randint(width - random_margin - 1, width - 1)
	y2 = random.randint(-random_margin, random_margin)
	x3 = random.randint(width - random_margin - 1, width - 1)
	y3 = random.randint(height - random_margin - 1, height - 1)
	x4 = random.randint(-random_margin, random_margin)
	y4 = random.randint(height - random_margin - 1, height - 1)

	dx1 = random.randint(-random_margin, random_margin)
	dy1 = random.randint(-random_margin, random_margin)
	dx2 = random.randint(width - random_margin - 1, width - 1)
	dy2 = random.randint(-random_margin, random_margin)
	dx3 = random.randint(width - random_margin - 1, width - 1)
	dy3 = random.randint(height - random_margin - 1, height - 1)
	dx4 = random.randint(-random_margin, random_margin)
	dy4 = random.randint(height - random_margin - 1, height - 1)

	pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
	pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
	M_warp = cv2.getPerspectiveTransform(pts1, pts2)
	img_warp = cv2.warpPerspective(img, M_warp, (width, height))
	return M_warp, img_warp


M_warp, img_warp = random_warp(img_ori, img_ori.shape[0], img_ori.shape[1])
print(M_warp)
my_show(img_warp)

# 12.图片腐蚀img erode
img_libai = cv2.imread("libai.png", 1)
erode_img = cv2.erode(img_libai, None, iterations=1)    # iterations是迭代次数
my_show(erode_img)

# 13图片膨胀img dilate
img_li = cv2.imread("libai.png", 1)
dilate_img = cv2.dilate(img_li, None, iterations=1)
my_show(dilate_img)

