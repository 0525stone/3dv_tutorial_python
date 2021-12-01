"""
reconstructing from 2d points to 3D points by several images by finding Fundamental matrix

# 출처 ; https://www.geeksforgeeks.org/feature-matching-using-brute-force-in-opencv/

data : 직접 구할 필요가 있음...

keyword : camera calibration, 3d reconstruction
"""
# Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np


# fundamental part
R = [[1,0,0],[0,0,-1],[0,1,0]]
t = [100,200,300]


################################################

# importing openCV library
import cv2


# function to read the images by taking there path
def read_image(path1, path2):
    read_img1 = cv2.imread(path1)
    read_img2 = cv2.imread(path2)
    read_img1 = cv2.pyrDown(read_img1)
    read_img2 = cv2.pyrDown(read_img2)
    read_img1 = cv2.pyrDown(read_img1)
    read_img2 = cv2.pyrDown(read_img2)

    return (read_img1, read_img2)


# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1, pic2):
    gray_img1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    return (gray_img1, gray_img2)


# function to detect the features by finding key points
# and descriptors from the image
def detector(image1, image2):
    # creating ORB detector
    detect = cv2.ORB_create()

    # finding key points and descriptors of both images using
    # detectAndCompute() function
    key_point1, descrip1 = detect.detectAndCompute(image1, None)
    key_point2, descrip2 = detect.detectAndCompute(image2, None)
    return (key_point1, descrip1, key_point2, descrip2)


# function to find best detected features using brute force
# matcher and match them according to there humming distance
def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    no_of_matches = brute_force.match(des1, des2)

    # finding the humming distance of the matches and sorting them
    no_of_matches = sorted(no_of_matches, key=lambda x: x.distance) # hamming distance의 순서별로
    return no_of_matches


# function displaying the output image with the feature matching
def display_output(pic1, kpt1, pic2, kpt2, best_match):
    # drawing the feature matches using drawMatches() function
    output_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)

    cv2.imwrite('save_.jpg',output_image)
    cv2.imshow('Output image', output_image)
    # plt.imshow(output_image)
    # plt.show()


# main function
if __name__ == '__main__':
    # giving the path of both of the images
    first_image_path = '../data/s10.jpeg'
    second_image_path = '../data/s12.jpeg'

    # reading the image from there paths
    img1, img2 = read_image(first_image_path, second_image_path)

    # converting the readed images into the gray scale images
    gray_pic1, gray_pic2 = convert_to_grayscale(img1, img2)

    # storing the finded key points and descriptors of both of the images
    key_pt1, descrip1, key_pt2, descrip2 = detector(gray_pic1, gray_pic2)

    # sorting the number of best matches obtained from brute force matcher
    number_of_matches = BF_FeatureMatcher(descrip1, descrip2)
    tot_feature_matches = len(number_of_matches)
    print(f'Total Number of Features matches found are {tot_feature_matches}')

    # after drawing the feature matches displaying the output image
    display_output(gray_pic1, key_pt1, gray_pic2, key_pt2, number_of_matches)

    # 확인해볼 지표
    print(f'key point : {type(key_pt1)}, len : {len(key_pt1)}, len : {len(key_pt2)}')

    # find fundamental matrix
    # opencv api : https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a
    # funda, mask = cv2.findFundamentalMat(np.array(key_pt1), np.array(key_pt2), cv2.FM_RANSAC,3,0.99,10)
    # print(funda)

    cv2.waitKey()
    cv2.destroyAllWindows()




