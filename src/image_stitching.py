"""
3D vision tutorial
https://github.com/sunglok/3dv_tutorial/blob/master/src/image_stitching.cpp
최박사님 것으로 진행하며 파이썬으로 영상처리하는 것에 익숙해지


"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


img1_dir = '../data/sample4.jpeg'
img2_dir = '../data/sample5.jpeg'

img1 = cv2.imread(img1_dir)
img2 = cv2.imread(img2_dir)

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



# example code from https://docs.opencv.org/4.5.2/d1/de0/tutorial_py_feature_homography.html
MIN_MATCH_COUNT = 10 # 10개도 matching 된게 없으면 지운단 얘

if len(matches)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
print(type(matches))
print(f'length comparison\nkeypoints1 {len(keypoints1)}\nmatches {len(matches)}')
# img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches,None,**draw_params) # match 전체를 그릴 때

# Draw first 15 matches
img3 = cv2.drawMatches(img1 = img1,
                        keypoints1 = keypoints1,
                        img2 = img2,
                        keypoints2 = keypoints2,
                        matches1to2 = matches[:15],
                        outImg = None,
                        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
H,W,C = img1.shape
merge = cv2.warpPerspective(img1,M,(W*2,H)) # img
merge2 = cv2.warpPerspective(img2,M,(W*2,H)) # img
img4 = cv2.hconcat([merge,img2])
img_ = cv2.hconcat([img1,img2])



img_out = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
plt.imshow(img_out)
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.close()









# fig = plt.figure(figsize=(10,10))
# plt.title('BRISK Matching')
# img_ = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
#
# plt.imshow(img_)
# plt.show()
# close plot
# plt.waitforbuttonpress(0) # this will wait for indefinite time
# plt.close(img3)



# # visualize
# img12 = cv2.hconcat([img1,img2])
# img_ = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
# plt.imshow(img_) #, cv2.COLOR_BGRA2RGBA)
# plt.show()

