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

    # top-15 matches check
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    # cv2.imshow('Output image', output_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

 # Fundamental matrices
    # extract points
    pts1 = []
    pts2 = []
    for i, (m) in enumerate(matches):
        if m.distance < 7: # 20
            # print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

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
    # result = cv2.hconcat([homography,homography2])

    cv2.imwrite('homography.png', homography)
    cv2.imshow("Homography", homography)
    # cv2.imshow("Homography", result)
    cv2.waitKey()
    cv2.destroyAllWindows()


# fundamental matrix
    # A = [[1,2],[3,4]]
    # B = [1,7]
    # print(A)
    # print(np.dot(A,B))
    # print(np.matmul(A, B))
    #
    # # match 된것 몇 개만 옮겨서 circle로 찍어보기
    # print(f'matches count : {len(matches)}')
    # for idx, p in enumerate(matches[:10]):
    #     # print(f'kp {type(kp1)}, p {type(p)}')  # kp1 : tuple, p : cv2.DMatch
    #     # print(f'point idx : {p.trainIdx}')
    #     # print(f'point {kp1[p.queryIdx]}')
    #     print(f'kp1 green {kp1[p.queryIdx].pt}')
    #     print(f'kp2 red {kp2[p.trainIdx].pt}')
    #
    #     x1 , y1 = kp1[p.queryIdx].pt
    #     x2, y2 = kp2[p.trainIdx].pt
    #
    # # # Matrix 형태1 : [[행][행]]
    # #     x1n = int(F_inv[0][0] * x1 + F_inv[0][1] * y1 + F_inv[0][2])
    #     # y1n = int(F_inv[1][0] * x1 + F_inv[1][1] * y1 + F_inv[1][2])
    #     # x2n = int(F_inv[0][0] * x2 + F_inv[0][1] * y2 + F_inv[0][2])
    #     # y2n = int(F_inv[1][0] * x2 + F_inv[1][1] * y2 + F_inv[1][2])
    # # Matrix 형태2 : [[열][열]] -> 이게 맞음
    #     x1n = int(F_inv[0][0] * x1 + F_inv[1][0] * y1 + F_inv[2][0])
    #     y1n = int(F_inv[0][1] * x1 + F_inv[1][1] * y1 + F_inv[2][1])
    #     x2n = int(F_inv[0][0] * x2 + F_inv[1][0] * y2 + F_inv[2][0])
    #     y2n = int(F_inv[0][1] * x2 + F_inv[1][1] * y2 + F_inv[2][1])
    #
    #     x1ni = int(F[0][0] * x1 + F[1][0] * y1 + F[2][0])
    #     y1ni = int(F[0][1] * x1 + F[1][1] * y1 + F[2][1])
    #     x2ni = int(F[0][0] * x2 + F[1][0] * y2 + F[2][0])
    #     y2ni = int(F[0][1] * x2 + F[1][1] * y2 + F[2][1])
    #
    #     x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
    #
    #     print(f'x1 y1 x2 y2 {x1} {y1} {x2} {y2}')
    #     print(f'inv x1 y1 x2 y2 {x1n} {y1n} {x2n} {y2n}')
    #     print(f'f x1 y1 x2 y2 {x1ni} {y1ni} {x2ni} {y2ni}')
    #
    #     plt.plot(x1, y1, color='green', marker='o')
    #     plt.plot(x2, y2, color='red', marker='o')
    #     plt.plot([x1,x2], [y1,y2], color='blue')
    #     # plt.plot(x1n, y1n, color='green', marker='v')
    #     plt.plot(x2n, y2n, color='red', marker='v')




        # print('-'*20)
    # plt.show()

    return 0


def main():
    # read list of images form dir in sorted order
    image_dir = '/Users/mac/Documents/dinoRing/'
    file_list = sorted(glob.glob(image_dir + '*.png'))

    # compute F matrix between two images
    first_image_path = '../data/s09.jpeg'
    second_image_path = '../data/s10.jpeg'

    H = compute_homography_matrix(first_image_path, second_image_path)
    # print(f'Fundamental matrix\n{F}')
    # F_inv = np.linalg.inv(F)
    # F_inv = np.linalg.pinv(F)
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