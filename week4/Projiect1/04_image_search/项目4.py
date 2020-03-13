import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
import time


# Function 1：  计算聚类中心 构造视觉单词词典
def getClusterCentures(img_paths, num_words):
    sift_det = cv2.xfeatures2d.SIFT_create()
    des_list = []  # 特征描述
    des_matrix = np.zeros((1, 128))
    if img_paths != None:
        for path in img_paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kps, des = sift_det.detectAndCompute(gray, None)
            des_matrix = np.row_stack((des_matrix, des))
            des_list.append(des)
    else:
        raise ValueError('图片路径有问题')

    des_matrix = des_matrix[1:, :]  # the des matrix of sift

    kmeans = KMeans(n_clusters=num_words, random_state=33)
    kmeans.fit(des_matrix)
    centres = kmeans.cluster_centers_  # 视觉聚类中心

    return centres, des_list


# Function 2：获得单张图片的特征向量
def des2feature(des, num_words, centures):
    img_feature_vec = np.zeros((1, num_words), 'float32')
    for i in range(des.shape[0]):
        feature_k_rows = np.ones((num_words, 128), 'float32')
        feature = des[i]
        feature_k_rows = feature_k_rows * feature
        feature_k_rows = np.sum((feature_k_rows - centures) ** 2, axis=1)
        index = np.argmin(feature_k_rows)
        img_feature_vec[0][index] = img_feature_vec[0][index] + 1
    img_feature_vec = preprocessing.normalize(img_feature_vec, norm='l2')  #L2归一化
    return img_feature_vec


# Function 3：获取所有图片的特征向量
def get_all_features(des_list, num_words):
    allvec = np.zeros((len(des_list), num_words), 'float32')
    for i in range(len(des_list)):
        allvec[i] = des2feature(des=des_list[i], num_words=num_words ,centures=centres)
    return allvec


# Function 4：检索图像
def retrieval_img(img_path, img_dataset, centures, img_paths):
    # 显示最近邻的图像数目
    num_close = 3
    # 读取待检索的图像
    img = cv2.imread(img_path)
    # 将待检索的图像转换为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 建立SIFT生成器
    sift_det = cv2.xfeatures2d.SIFT_create()
    # 检测SIFT特征点，并计算描述子
    kp, des = sift_det.detectAndCompute(img, None)
    # 获得待检索的图像的特征向量
    feature = des2feature(des=des, num_words=num_words, centures=centures)
    # 找出与待检索的图像最像的num_close个图像
    sorted_index = getNearestImg(feature, img_dataset, num_close)
    # 显示最相似的图片集合
    showImg(img_path, sorted_index, img_paths)


# Function 5：找出与目标图像最像的num_close个图像
def getNearestImg(feature, dataset, num_close):
    features = np.ones((dataset.shape[0], len(feature)), 'float32')
    features = features * feature  # feature:目标图像特征
    dist = np.sum((features - dataset) ** 2, axis=1)
    dist_index = np.argsort(dist)  # 排序
    return dist_index[:num_close]  # 返回最相似的num_close个图像


# Function 6：显示最相似的图片集合
def showImg(target_img_path, index, dataset_paths):
    paths = []
    for i in index:
        paths.append(dataset_paths[i])

    plt.figure(figsize=(10, 20))  # 设置图片大小
    plt.subplot(432), plt.imshow(plt.imread(target_img_path)), plt.title('target_image')

    for i in range(len(index)):
        plt.subplot(4, 3, i + 4), plt.imshow(plt.imread(paths[i]))
    plt.show()


# Function 7：固定长宽比例 缩放图像
def my_resize(width, img):
    h, w, d = img.shape
    r = width / w
    dim = (width, int(h * r))
    resized = cv2.resize(img, dim)
    return resized


# Function 8 ： 无切边 旋转图像
def rotate_bound(image, angle):
    # 获得图像的尺寸，然后确定中心
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 获得旋转矩阵（负角度-顺时针旋转），然后获得正弦和余弦（即矩阵的旋转分量）
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵（考虑平移）
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # 执行实际旋转并返回图像
    return cv2.warpAffine(image, M, (nW, nH))


# Function 9 ：预处理图片
def preprocess(training_path_ori, imageType):
    if imageType == "target":
        image_pre = cv2.imread(training_path_ori)
        image_pre = my_resize(500, image_pre)
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #image_pre=cv2.filter2D(image_pre,-1,kernel_sharpen)
        cv2.imwrite("target.jpg", image_pre)
        return "target.jpg"
    elif imageType == "train":
        training_names_ori = os.listdir(training_path_ori)
        pic_names = ['bmp', 'jpg', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf',
                     'ufo', 'eps', 'ai', 'raw', 'WMF']
        for name in training_names_ori:
            file_format = name.split('.')[-1]
            if file_format not in pic_names:
                training_names_ori.remove(name)

        img_paths_ori = []  # 所有图片路径
        for name in training_names_ori:
            img_path_ori = os.path.join(training_path_ori, name)
            img_paths_ori.append(img_path_ori)
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        count = 1
        for name in img_paths_ori:
            image_pre = cv2.imread(name)
            image_pre = my_resize(500, image_pre)
            #image_pre=cv2.filter2D(image_pre,-1,kernel_sharpen)
            count = count + 1
            cv2.imwrite(str(count) + ".jpg", image_pre)

        training_path = r'D:\PycharmProjects\4'
        training_names = os.listdir(training_path)
        pic_names = ['jpg']
        for name in training_names:
            file_format = name.split('.')[-1]
            if file_format not in pic_names:
                training_names.remove(name)
        img_paths = []
        for name in training_names:
            img_path = os.path.join(training_path, name)
            img_paths.append(img_path)
        return img_paths


if __name__ == '__main__':
    # 训练图片的途径
    training_path_ori = r'D:\AI_CV\train_1'
    # 预处理图片
    img_paths = preprocess(training_path_ori, "train")

    num_words = 5  # 聚类中心数
    # 计算聚类中心 构造视觉单词词典
    centres, des_list = getClusterCentures(img_paths, num_words)

    # 获得所有训练集图片的特征向量
    img_features = get_all_features(des_list, num_words)

    # 待检索的图片的路径
    path = r'D:\AI_CV\image\dog.82.jpg'
    path = preprocess(path, imageType="target")

    # 检索图片
    retrieval_img(path, img_features, centres, img_paths)
    os.remove("target.jpg")
