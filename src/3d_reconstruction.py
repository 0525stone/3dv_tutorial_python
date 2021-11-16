"""
reconstructing from 2d points to 3D points by several images by finding Fundamental matrix

data : 직접 구할 필요가 있음...

keyword : camera calibration, 3d reconstruction
"""
# Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np


img1_dir = '../data/sample4.jpeg'
img2_dir = '../data/sample5.jpeg'

img1 = cv2.imread(img1_dir)
img2 = cv2.imread(img2_dir)

# fundamental part