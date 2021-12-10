"""
fundamental matrix 코드 기준으로 뒷부분만 변경

코드 예시 참조 : https://www.pythonpool.com/cv2-findhomography/

data summary

ss : no damage
s : yes damage on 우/전
(오른쪽 왼쪽은 영상을 바라보는 기준임)
정면 : ss01 ss06 s12
우/전 : ss02 ss03 ss04 ss05 s13 s14 s15 s16
좌/전 : ss07 s09 s10 s11
좌/후 : s01 s06 s07 s08
우/후 : s02 s03 s04 s05
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
    # img = cv2.pyrDown(img)
    # img = cv2.pyrDown(img)
    print(f'image size : {img.shape}')

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


def compute_homography_matrix(filename1, filename2):
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
    print(f'matches : {len(matches)}')

    # top-15 matches check
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    cv2.imshow('Output image', output_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

 # homogeneous matrices
    # extract points
    pts1 = []
    pts2 = []
    for i, (m) in enumerate(matches):
        if i<11:
        # if m.distance < 7: # 20
        #     print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    # # extract points
    # pts1 = []
    # pts2 = []
    # for i, (m) in enumerate(matches):
    #     if m.distance < 7: # 20
    #         print(m.distance)
    #         pts2.append(kp2[m.trainIdx].pt)
    #         pts1.append(kp1[m.queryIdx].pt)
    # pts1 = np.asarray(pts1)
    # pts2 = np.asarray(pts2)


    # # Compute fundamental matrix
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # F_inv = np.linalg.pinv(F)  # singular matrix error떠서 pinv로 변경

    # Find Homography matrix
       # pts2를 pts1로 변환하는 Homography matrix를 구해줌
    matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    h, w, d = img1.shape

    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # dst = cv2.perspectiveTransform(pts1, matrix)

    # homography = cv2.polylines(img1, [np.int32(dst)], True, (255, 0, 0), 3)
    homography = cv2.warpPerspective(img2, matrix, (w, h))
    # homography2 = cv2.warpPerspective(img1, matrix, (w, h))
    homography = cv2.hconcat([img2,homography])
    homography = cv2.hconcat([homography, img1])

    cv2.imwrite('homography.png', homography)
    cv2.imshow("Homography", homography)
    # cv2.imshow("Homography", result)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return 0


def main():
    # read list of images form dir in sorted order
    image_dir = '/Users/mac/Documents/dinoRing/'
    file_list = sorted(glob.glob(image_dir + '*.png'))

    # compute F matrix between two images
    first_image_path = '../data/check2_.png' # s13, s15 : 비교적 선명한 damage car
    second_image_path = '../data/check4_.png' # ss05

    H = compute_homography_matrix(first_image_path, second_image_path)


if __name__ == '__main__':
    main()