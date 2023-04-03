# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:27:52 2022
Microsoft Windows10 家庭中文版
版本20H2(操作系统内部版本19042.1586)
处理器 lntel(R) Core(TM) i5-8300H CPU @ 2.30GHz2.30 GHz
机带RAM 8.00 GB (7.80 GB可用)
GPU0 lntel(R) UHD Graphics 630
GPU1 NVIDIA GeForce GTX 1050 Ti
应用环境：python3.6+opencv3.4.1
调用cv2.xfeatures2d.SURF进行特征点检测、特征匹配
使用RANSAC方法滤去离群点，生成新的特征点对列表，再进行特征匹配与原始效果对比
借助cv2.findHomography计算由图B到图A的单应性矩阵求解
最终，借助cv2.warpPerspective函数实现将图B变换至图A坐标系下
并根据图片特征，以进行线性加权的方式进行融合，消除明显拼接边，裁剪掉黑色无用部分，得到理想效果
@author: 10554
"""

import cv2
import numpy as np
import random
import math


def compute_fundamental(x1, x2):
    #判定x1,x2是否大小匹配
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    # 建立方程矩阵
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]
    # 计算线性最小二乘解
    U, S, V = np.linalg.svd(A)
    # 约束F
    F = V[-1].reshape(3, 3)
    #将最后一个奇异值置零，使秩为2
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F / F[2, 2]


def compute_fundamental_normalized(x1, x2):
    """从对应点计算基本矩阵
   （x1，x2 3*n阵列）使用归一化8点算法。"""
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    # 规格化图像坐标
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])   #计算点到直线的距离与点组到直线的标准差的比值
    T1 = np.array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = np.dot(T1, x1)#计算矩阵乘积
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])  #计算点到直线的距离与点组到直线的标准差的比值
    T2 = np.array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = np.dot(T2, x2) #计算矩阵乘积
    # 用归一化坐标计算F
    F = compute_fundamental(x1, x2)
    #反向规范化
    F = np.dot(T1.T, np.dot(F, T2))
    return F / F[2, 2]


def randSeed(good, num=8):
    '''
    参数 good为初始的匹配点对
    参数 num为选择随机选取的点对数量
    函数返回 8个点对的列表
    '''
    eight_point = random.sample(good, num)
    #在点对good中选取随机的8点
    return eight_point


def PointCoordinates(eight_points, keypoints1, keypoints2):
    '''
    参数eight_points为随机八点
    参数 keypoints1为图A的对应点坐标
    参数 keypoints2为图A的对应点坐标
    函数返回8个点
    '''
    x1 = []
    x2 = []
    tuple_dim = (1.,)
    for i in eight_points:
        tuple_x1 = keypoints1[i.queryIdx].pt + tuple_dim #queryIdx特征点序列是图片A中的
        tuple_x2 = keypoints2[i.trainIdx].pt + tuple_dim #trainIdx特征点序列是图片B中的
        x1.append(tuple_x1)
        x2.append(tuple_x2)                              #将处理后的特征点加入新列表

    return np.array(x1, dtype=float), np.array(x2, dtype=float)


def ransac(good, keypoints1, keypoints2, confidence, iter_num):
    #初始化匹配点数量
    Max_num = 0
    #初始化矩阵
    good_F = np.zeros([3, 3])
    #初始化点对矩阵
    inlier_points = []
    for i in range(iter_num):

        eight_points = randSeed(good)#取随机八点
        x1, x2 = PointCoordinates(eight_points, keypoints1, keypoints2) #得到对应的点坐标
        F = compute_fundamental_normalized(x1.T, x2.T)                  #将点转置后进行归一化处理
        num, ransac_good = inlier(F, good, keypoints1, keypoints2, confidence)

        if num > Max_num:                    #若最终筛选后点云数量大于设定的最大值，则以筛选后的数量为准
            Max_num = num                    #将处理后的数据、列表整合为输出结果
            good_F = F
            inlier_points = ransac_good

    return Max_num, good_F, inlier_points


def computeReprojError(x1, x2, F):
    """
    计算投影误差
    """
    ww = 1.0 / (F[2, 0] * x1[0] + F[2, 1] * x1[1] + F[2, 2])
    dx = (F[0, 0] * x1[0] + F[0, 1] * x1[1] + F[0, 2]) * ww - x2[0]
    dy = (F[1, 0] * x1[0] + F[1, 1] * x1[1] + F[1, 2]) * ww - x2[1]

    return dx * dx + dy * dy


def inlier(F, good, keypoints1, keypoints2, confidence):
    num = 0
    ransac_good = []
    x1, x2 = PointCoordinates(good, keypoints1, keypoints2)
    for i in range(len(x2)):
        line = F.dot(x1[i].T)  #求good_F及x1[i]转置矩阵的内积
        # 在对极几何中极线表达式为[A B C],Ax+By+C=0,  方向向量可以表示为[-B,A]
        line_v = np.array([-line[1], line[0]])
        err = h = np.linalg.norm(np.cross(x2[i, :2], line_v) / np.linalg.norm(line_v))
        if abs(err) < confidence:      #若误差在阈值范围内，则将该点对加入集和，并在数量上+1
            ransac_good.append(good[i])
            num += 1
    return num, ransac_good             #返回点对数量，即经ransac方法处理后的特征点对列表

if __name__ == '__main__':

    #读取原图像
    psd_img_1 = cv2.imread(r'./hw4/A.png', cv2.IMREAD_COLOR)
    psd_img_2 = cv2.imread(r'./hw4/B.png', cv2.IMREAD_COLOR)

    # 1) SURF特征计算
    surf = cv2.xfeatures2d.SURF_create()
    # 使用SURF查找关键点和描述符
    kp1, des1 = surf.detectAndCompute(psd_img_1, None)
    kp2, des2 = surf.detectAndCompute(psd_img_2, None)
    #针对SURF算法,建立DescriptorMatcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(des1, des2)
    #使用RANSAC滤除离群点
    Max_num, good_F, inlier_points = ransac(matches, kp1, kp2, confidence=30, iter_num=500)
    #进行特征点匹配，用绿色线条画出两张图对应的匹配点，img3为原始效果,img4为利用ransac优化后效果
    img3 = cv2.drawMatches(psd_img_1, kp1, psd_img_2, kp2, matches, None,[0,255,0], flags=2)
    img4 = cv2.drawMatches(psd_img_1, kp1, psd_img_2, kp2, inlier_points, None,[0,255,0], flags=2)
    # 获取关键点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # 测试图像的关键点的坐标位置
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # 样本图像的关键点的坐标位置
    #根据对应点计算单应性矩阵（图B到图A）,返回值中M为变换矩阵。mask是掩模,online的点
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    data1 = [[round(j,3) for j in M[i]] for i in range(len(M))]
    print(data1)
    #提取目标图片和要转换图片的大小信息，为下一步合并准备（长度宽度要匹配）
    h, w = psd_img_2.shape[0], psd_img_1.shape[1]
    #根据计算的单应矩阵把图B变换到图A的坐标系下
    wrap = cv2.warpPerspective(psd_img_2, M, (math.floor(2.5 * w), h))
    #与图A以线性加权的方式进行融合
    for h_i in range(h):
        cur_l = -1
        cur_r = -1
        for w_i in range(w):
            #在重叠区范围线性加权融合图像
            if sum(wrap[h_i][w_i]) / 3 > 0.001 and sum(psd_img_1[h_i][w_i]) / 3 > 0.001:
                if cur_l < 0 and cur_r < 0:
                    cur_l = w_i
                elif cur_l > 0:
                    cur_r = w_i
            if sum(wrap[h_i][w_i]) / 3 > 0.001 and sum(psd_img_1[h_i][w_i]) / 3 < 0.001:
                if cur_l > 0 and cur_r > 0:
                    break
        ler2l = cur_r - cur_l
        for w_i in range(w):
            if w_i < cur_l:
                wrap[h_i][w_i] = psd_img_1[h_i][w_i]#小于这一边界，则将图片复制到指定区域
            elif w_i >= cur_l and w_i < cur_r:
                f_l = (w_i - cur_l) / ler2l         #若在重叠区域内，则根据像素的差值线性加权
                f_r = (cur_r - w_i) / ler2l         #从而消除明显的边界线，平滑过度
                wrap[h_i][w_i] = wrap[h_i][w_i] * f_l + psd_img_1[h_i][w_i] * f_r
    
    #去除黑色无用部分：检验像素为0的黑色部分，将图片裁剪至合适大小  
    rows, cols = np.where(wrap[:,:,0] !=0)         #选出图片中非0区域
    min_row, max_row = min(rows), max(rows) +1     #确定行的范围(高度h)
    min_col, max_col = min(cols), max(cols) +1     #确定列的范围(宽度w)
    wrap = wrap[min_row:max_row,min_col:max_col,:] #降黑色部分裁剪掉
    
    #存储各步骤图片
    cv2.imshow("image1", img3)
    cv2.imwrite(r'./output/SURF_match.png',img3)
    cv2.imshow("image2", img4)
    cv2.imwrite(r'./output/SURF_ransac.png',img4)
    cv2.imshow('result', wrap)
    cv2.imwrite(r'./output/SURF_stitch.png',wrap)
    cv2.waitKey(0)           # 等待按键按下
    cv2.destroyAllWindows()  # 清除所有窗口

