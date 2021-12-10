"""
find contour and make bounding box then get IoU


"""
import cv2

import os
import cv2
from shapely.geometry import Polygon
import numpy as np

# from google.colab.patches import cv2_imshow # for colab
# import geopandas as gpd

def get_iou(img1, img2):
  # print(f'img1 shape : {img1.shape}')
  # print(f'img2 shape : {img2.shape}')
  # print(f'img1 type : {type(img1)}')
  # print(f'img1 type : {type(img1)}')
  cv2.imshow('result', img1*img2)
  return len(np.where(img1*img2 !=0)[0])

def check_overlap(img_dir):
  img = cv2.imread(img_dir) #파일명
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백사진으로 변환

# find pixel value to divid mask per classes
  gray_np = np.array(gray)
  pix = np.unique(gray)
  # print(f'check pixel value {pix}')

# 각 class 별로 binary mask 분리해주는 부분(binary 변수에 각 저장)
  w,h=gray.shape
  binary = np.zeros((2,w,h))
  # print(f'size : {binary.shape}')
  for i in range(len(pix)-1):
    ret1, binary1 = cv2.threshold(gray,pix[i] , 255, 0)
    ret2, binary2 = cv2.threshold(gray,pix[i+1],255, 0) # binary2 가 자동차
    temp = binary1- binary2
    binary[i] = temp

  # print('-'*50)
  # print(f'binary 확인 : {len(binary)}')
  # print(f'binary type 확인 : {type(binary[1])}')
  # print(f'temp type 확인 : {type(temp)}')
  # print(f'binary shape 확인 : {binary[1].shape}')
  # print(f'temp shape 확인 : {temp.shape}')

# 무조건 도로를 먼저 찾게 되므로 binary의 마지막 항이 자동차임
  # binary[1], temp 두 개가 완벽하게 똑같은데 binary[1]은 에러가 뜸(dtype이라고 하면서... numpy array로 할당하면서 타입이 바뀌나?)
  contours, h = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # contours, h = cv2.findContours(binary[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  x_min = contours[0][:,:,0].min()
  x_max = contours[0][:,:,0].max()
  y_min = contours[0][:,:,1].min()
  y_max = contours[0][:,:,1].max()
  bbox = np.zeros(gray.shape)
  cv2.rectangle(bbox, (x_min, y_min), (x_max, y_max), (123, 0, 0), -1)
  cv2.imshow('car bbox', bbox)
  cv2.imshow('road',binary[0])

# iou 계산해보기
  iou = get_iou(bbox, binary[0])

  if iou!=0:
    print(f"iou is not zero. overlapped. car on the road {iou}")
  else:
    print("iou is zero. no car on the road")

  cv2.waitKey(0)
  cv2.destroyAllWindows()


def main():

  img_dir = '../data/l01.png'
  check_overlap(img_dir)


if __name__ == '__main__':
  main()