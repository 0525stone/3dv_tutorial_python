"""
Blur detection
- opencv
- input image
- crop image
- get score
- output boolean_check(threshold로), crop_img

Memo from notes
- 문제 : 번호판 부분이 blur 되어 있는 것에 대한 지표를 뽑아야함
    - Canny는 blur 되어 있는 번호판 부분에 대해서 edge를 하나도 잡지 않음
    - Canny 결과의 variance를 확인해 본 결과, edge가 없는 것과 있는 경우에 대해서 값 차이가 크게 남(그렇기 때문에 배경에서 edge 많이 잡히면 오동작 위험)
- 해결 : 입력 영상에서 차량 번호판이 있을 만한 위치에 대해서 crop을 하고, crop한 영상에 대해서 canny 결과의 variance를 확인 후, 임계치와 비교하여 blur 여부 결정
  - Canny, Laplacian, Sobel 중에서 Canny가 제일 동작을 잘하는 것 확인
    - Laplacian : 2차 미분 필터
    - Sobel : 1차 미분 필터
    - Canny : sobel filter + hyteresis thresholding 으로 에지 여부 판단 -> 결과 제일 명확하게 나옴
"""
from pandas import DataFrame
from matplotlib import pyplot as plt

import pandas as pd
import os
import cv2


def score_blur(image, threshold=1000):
# cropped image
    w = image.shape[1]
    crop_image = image[: , int(w/3): int(2*w/3)]

    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    score = cv2.Canny(gray, 100, 255).var()

    if(score>threshold):
        return True, crop_image
    else:
        return False, crop_image

# print('test for score_blur')
#
# file_dir = '../data/side/scratch'
# file_list = os.listdir(file_dir)
# print(f'files : {len(file_list)}, {type(file_list)}')
#
# import random
# r_seed = 25
# n_sample = 10
#
# random.seed(r_seed)
# r_sample = random.sample(file_list, n_sample)
#
# print(r_sample)
#
# for samplename in r_sample:
#     img_dir = os.path.join(file_dir, samplename)
#     img = cv2.imread(img_dir)
#
#     _bool, img = score_blur(img)
#     cv2.imshow('check',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


