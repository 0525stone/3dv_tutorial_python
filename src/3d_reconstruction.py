"""
reconstructing from 2d points to 3D points by several images by finding Fundamental matrix

data : 직접 구할 필요가 있음...

keyword : camera calibration, 3d reconstruction
"""
# Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np


img1_dir = '../data/a1.jpeg'
img2_dir = '../data/a2.jpeg'

img1 = cv2.imread(img1_dir)
img2 = cv2.imread(img2_dir)

# fundamental part

import numpy as np

R = [[1,0,0],[0,0,-1],[0,1,0]]
t = [100,200,300]

# Keypoint detector
BRISK = cv2.BRISK_create()
# Find the keypoints and compute the descriptors for input and training-set image
keypoints1, descriptors1 = BRISK.detectAndCompute(img1, None)
keypoints2, descriptors2 = BRISK.detectAndCompute(img2, None)

# create BFMatcher object
BFMatcher1 = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
                         crossCheck = True)

# Matching descriptor vectors using Brute Force Matcher
matches = BFMatcher1.match(queryDescriptors = descriptors1,
                          trainDescriptors = descriptors2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)


