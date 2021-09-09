"""
3D vision tutorial
https://github.com/sunglok/3dv_tutorial/blob/master/src/image_stitching.cpp
최박사님 것으로 진행하며 파이썬으로 영상처리하는 것에 익숙해지


"""

import cv2
import matplotlib.pyplot as plt


img1_dir = '../data/sample1.jpeg'
img2_dir = '../data/sample2.jpeg'

img1 = cv2.imread(img1_dir)
img2 = cv2.imread(img2_dir)

# Keypoint detector
BRISK = cv2.BRISK_create()
# Find the keypoints and compute the descriptors for input and training-set image
keypoints1, descriptors1 = BRISK.detectAndCompute(img1, None)
keypoints2, descriptors2 = BRISK.detectAndCompute(img2, None)

# create BFMatcher object
BFMatcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
                         crossCheck = True)

# Matching descriptor vectors using Brute Force Matcher
matches = BFMatcher.match(queryDescriptors = descriptors1,
                          trainDescriptors = descriptors2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 15 matches
output = cv2.drawMatches(img1 = img1,
                        keypoints1 = keypoints1,
                        img2 = img2,
                        keypoints2 = keypoints2,
                        matches1to2 = matches[:15],
                        outImg = None,
                        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(10,10))
plt.title('BRISK Matching')
plt.imshow(output)
plt.show()

# # visualize
# img12 = cv2.hconcat([img1,img2])
# img_ = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
# plt.imshow(img_) #, cv2.COLOR_BGRA2RGBA)
# plt.show()

