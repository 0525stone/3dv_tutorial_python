"""
출처 : https://www.programcreek.com/python/?project_name=PacktPublishing%2FPractical-Computer-Vision#
programcreek 에 좋은 소스 많음
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

print(cv2.__version__)
import glob


# With jupyter notebook uncomment below line
# %matplotlib inline 
# This plots figures inside the notebook


def compute_orb_keypoints(filename):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img = cv2.imread(filename)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)

    # create orb object
    orb = cv2.ORB_create()

    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)

    # detect keypoints
    kp = orb.detect(img, None)

    # for detected keypoints compute descriptors. 
    kp, des = orb.compute(img, kp)

    return img, kp, des


def brute_force_matcher(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def compute_fundamental_matrix(filename1, filename2):
    """
    Takes in filenames of two input images 
    Return Fundamental matrix computes 
    using 8 point algorithm
    """
    # compute ORB keypoints and descriptor for each image
    img1, kp1, des1 = compute_orb_keypoints(filename1)
    img2, kp2, des2 = compute_orb_keypoints(filename2)

    # compute keypoint matches using descriptor
    matches = brute_force_matcher(des1, des2)

    #
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], None, flags=2)
    cv2.imshow('Output image', output_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

 # Fundamental matrices
    # extract points
    pts1 = []
    pts2 = []
    for i, (m) in enumerate(matches):
        if m.distance < 20:
            # print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)


    # match 된것 몇 개만 옮겨서 circle로 찍어보기
    print(f'matches count : {len(matches)}')
    for idx, p in enumerate(matches[:10]):
        # print(f'kp {type(kp1)}, p {type(p)}')  # kp1 : tuple, p : cv2.DMatch
        # print(f'point idx : {p.trainIdx}')
        # print(f'point {kp1[p.queryIdx]}')
        print(f'kp1 green {kp1[p.queryIdx].pt}')
        print(f'kp2 red {kp2[p.trainIdx].pt}')

        x1 , y1 = kp1[p.queryIdx].pt
        x2, y2 = kp2[p.trainIdx].pt
        x1n = int(F[0][0] * x1 + F[1][0] * y1 + F[2][0])
        y1n = int(F[0][1] * x1 + F[1][1] * y1 + F[2][1])
        x2n = int(F[0][0] * x2 + F[1][0] * y2 + F[2][0])
        y2n = int(F[0][1] * x2 + F[1][1] * y2 + F[2][1])

        x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
        plt.plot(x1, y1, color='green', marker='o')
        plt.plot(x2, y2, color='red', marker='o')
        plt.plot([x1,x2], [y1,y2], color='blue')
        plt.plot(x1n, y1n, color='green', marker='v')
        plt.plot(x2n, y2n, color='red', marker='v')




        print('-'*20)
    plt.show()

    return F


def main():
    # read list of images form dir in sorted order 
    image_dir = '/Users/mac/Documents/dinoRing/'
    file_list = sorted(glob.glob(image_dir + '*.png'))

    # compute F matrix between two images
    first_image_path = '../data/s09.jpeg'
    second_image_path = '../data/s10.jpeg'
    # print(compute_fundamental_matrix(first_image_path, second_image_path))
    F = compute_fundamental_matrix(first_image_path, second_image_path)
    # print(f'Fundamental matrix\n{F}')
    F_inv = np.linalg.inv(F)
    # print(f'Fundamental inv mat\n{F_inv}')
    # print(f'matmul F, F_inv\n{np.matmul(F, F_inv)}')
    # print(f'matmul F_inv, F\n{np.matmul(F_inv, F)}')
    # print(f'dot F, F_inv\n{np.dot(F, F_inv)}')
    # print(f'dot F_inv, F\n{np.dot(F_inv, F)}')

    # # numpy inverse matrix 동작 확인
    # print('-'*50)
    # arr = np.array([
    #     [[1,2,3],
    #      [0,4,5],
    #      [1,0,6]]])
    #
    # arr_inv = np.linalg.inv(arr)
    #
    # print(arr_inv)
    # print(np.dot(arr_inv,arr))



if __name__ == '__main__':
    main()