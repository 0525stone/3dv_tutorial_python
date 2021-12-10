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


sample_dir = '../data/l01.png'
# sample_dir = '../data/check1_.png'

# os.chdir('C:/Users/user/Desktop/hehe/') #폴더 경로
img = cv2.imread(sample_dir) #파일명

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백사진으로 변환

gray_np = np.array(gray)
ch = np.unique(gray)
print(f'check {ch}')

ret, binary = cv2.threshold(gray, 10, 255, 0)
cv2.imshow('checks',binary)
contours, h = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(f'contours {contours}')

# segmentation label 뽑아주는 코드
line_tot = []
ex_line=[]
if len(contours):  # contour 하나라도 존재하면
  for contour in contours:
    ex_line = [[p[0],p[1]] for p in contours]
    print(ex_line)
    line_tot.append([ex_line])


image = cv2.drawContours(img, contours, -1, (0,0,255), 2)

# cv2.imshow('contours',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.drawContours 로 contour 구하고 각 mask 구함


print('done')
