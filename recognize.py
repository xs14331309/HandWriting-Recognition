import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.externals import joblib

dim = 28
digits = []
normal_images = []

i_img = 1 #index

path_test_image = 'hough/hough_%d.bmp' % i_img
image_color = cv2.imread(path_test_image)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
img_height = image.shape[0]
img_width = image.shape[1]
if img_height > 1500:
    image = cv2.resize(image, (int(img_width / 3), int(img_width / 3)))
    img_height = image.shape[0]
    img_width = image.shape[1]
image = cv2.GaussianBlur(image, (3, 3), 0) # 高斯模糊
image = image[5:img_height - 5, 5:img_width - 5] # 去除边框噪点

cv2.namedWindow("Gray")
cv2.imshow("Gray", image)
cv2.waitKey(0)

# 二值化
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 10)
cv2.imshow('binary image', adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 垂直方向像素统计
horizontal_sum = np.sum(adaptive_threshold, axis=1)
# plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
# plt.gca().invert_yaxis()
# plt.show()

################################
# 按行切割
# array_val:img
# minimun_val: 最小阈值
# minimun_range: 最小范围
# row: if true 按行切割 or false 按列切割
# rate: 重叠字符切割比率
# 行最大宽度
#################################
def extract_peek_ranges_from_array(array_vals, minimun_val=20, minimun_range=40, row=True, rate=1.5, max_range=45):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    if row:
        min_peek_range = min([end - start for (start, end) in peek_ranges])
        for i, peek_range in enumerate(peek_ranges):
            if (peek_range[1] - peek_range[0] > rate * min_peek_range):
                start = peek_range[0]
                end = peek_range[1]
                #mid = int((start + end) / 2)
                mid = peek_range[0] + max_range
                peek_ranges.remove(peek_range)
                peek_ranges.insert(i, (start, mid))
                peek_ranges.insert(i + 1, (mid, end))
    return peek_ranges

#
def median_split_ranges(peek_ranges, rate=1):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i] / (rate * median_w), 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges

'''
1: (20, 40, True, 1.5, 45)
2: (20, 40, True, 1.5, 45)
3: (20, 40, True, 1.5, 45)
4: (20, 40, True, 1.2, 80)
5: (20, 40, True, 1.5, 45)
'''
#　找出文本行
peek_ranges = extract_peek_ranges_from_array(
    horizontal_sum, 20, 40, True, 1.5, 45)
print(peek_ranges)
# 去除多余的文本行
if i_img == 4:
    peek_ranges.pop(1)
    peek_ranges.pop(2)
if i_img == 5:
    peek_ranges.pop(1)

line_seg_adaptive_threshold = np.copy(adaptive_threshold)
for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    w = line_seg_adaptive_threshold.shape[1]
    h = peek_range[1] - y
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
cv2.imshow('line image', line_seg_adaptive_threshold)
cv2.waitKey(0)


'''
image1 (_, 20, 5, False)
image2 (_, 700, 5, False)
image3 (_, 800, 5, False)
image4 (_, 500, 5, False)
image5 (_, 500, 5, False)
'''

# 逐行提取
vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    vertical_sum = np.sum(line_img, axis=0)
    vertical_peek_ranges = extract_peek_ranges_from_array(
        vertical_sum,
        minimun_val=20,
        minimun_range=5,
        row=False)
    vertical_peek_ranges = median_split_ranges(vertical_peek_ranges, 1.2)
    vertical_peek_ranges2d.append(vertical_peek_ranges)


# Draw
color = (0, 0, 255)
for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        copyImage = adaptive_threshold[y:y + h, x:x + w]
        x, y, w, h = cv2.boundingRect(copyImage)
        digits.append(copyImage[y:y + h, x:x + w])
        cv2.rectangle(image, pt1, pt2, color)
        # cv2.imshow('digit image', copyImage)
        # cv2.waitKey(0)
        # cv2.destroyWindow('digit image')

cv2.imshow('char image', image)
cv2.waitKey(0)

for digitImage in digits:
    print(digitImage.shape)
    (rh, rw) = digitImage.shape
    if rh >= rw:
        scale = rh / dim
        w = int(rw / scale)
        shape = (w, dim)
        digitImage = cv2.resize(digitImage, shape)
        img = np.zeros((dim, dim), np.uint8)
        begin = int((dim - w) / 2)
        img[0:dim, begin:begin + w] = digitImage
    else:
        scale = rw / dim
        h = int(rh / scale)
        shape = (dim, h)
        digitImage = cv2.resize(digitImage, shape)
        img = np.zeros((dim, dim), np.uint8)
        begin = int((dim - h) / 2)
        img[begin:begin + h, 0:dim] = digitImage
    cv2.imshow('digit image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('digit image')
    img = np.round(np.divide(np.reshape(img, (784, )), 255))
    # print(img)
    normal_images.append(img)
'''
# 预测
normal_images = pd.DataFrame(normal_images)
print(len(normal_images))

model = xgb.Booster(model_file='xgb.model')
test = xgb.DMatrix(normal_images)
predict = model.predict(test)
print(predict)
'''
