"""
출처 : https://gist.github.com/Pawe1/0e0b2fe4010e4ac6275e1aa1381ddaca

실험 기록 위하여 엑셀 시트 접근, 데이터 기록 파트 추가
"""
from imutils import paths
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot as plt

import os
import cv2
import random
import argparse


def image_read(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.pyrDown(image,cv2.IMREAD_GRAYSCALE)
    # image = cv2.pyrDown(image)
    # image = cv2.pyrDown(image)
    return image


def variance_of_laplacian(image, flag = 0):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    # if()

	return cv2.Laplacian(image, cv2.CV_64F).var()


def sobel_filter(gray_img):
    sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3)
    # fm = variance_of_laplacian(sobel)
    fm = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3).var()
    h, w = gray_img.shape
    # print(f'size : {gray_img.shape}')
    # cv2.putText(sobel, "Not blurry: {:.2f}".format(fm), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    return sobel, fm


def record_results(filename, data, sheetname):
    """
    merge cells in excel by python : https://developer-ankiwoong.tistory.com/512

    img_list
    1. get image list by os.listdir
    2. get image list by img_[class].txt file

    need to deal with excel
    - https://stackoverflow.com/questions/30294661/pandas-save-dataframe-to-open-workbook
    - https://book.coalastudy.com/data_crawling/week4/stage3
    """
    data_root = '../data/side'
    _cls = ['good','scratch','blur']
    sheetname='exp1'
    filename = 'experiment_side.xlsx'

# file이 있으면 append하고 없으면 새로 만들어서 하게 끔
    if os.path.isfile(filename):
        print('file exists')
        writer = pd.ExcelFile(filename)
        writer.sheet_names
        print(f'sheet name : {writer.sheet_names}')
    else:
        writer = pd.ExcelWriter(filename)

    for c in _cls:
        img_dir = os.path.join(data_root, c)
        img_list = os.listdir(img_dir) # text file 로도 불러올 수 있음
        img_list = sorted(img_list)
        scores = {}

        for idx, img_path in enumerate(img_list,1):
            img_ = cv2.imread(os.path.join(data_root,c,img_path))
            # print(os.path.join(data_root,c,img_path))

            img = cv2.pyrDown(img_)
            # img = cv2.pyrDown(img)
            # img = cv2.pyrDown(img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray) # score
            # print(f'{type(fm)},  {fm}')
            # print(f'{type(imgexp1_)},  {img_}')
            scores[idx] = {c : {'img': img_path, 'score': fm}}

            text = 'Not blurry '
            cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)


            # cv2.imshow("Image", img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        # class 하나 끝날 때마다 엑셀 추가
        scores_df = DataFrame(scores).T
        scores_df.to_excel(writer, sheet_name=sheetname)

    writer.save()


image_root = '../data/side'# os.path.join(home_dir, )
image_dir = os.path.join(image_root, 'good')
blur_dir = os.path.join(image_root, 'blur')
scratch_dir = os.path.join(image_root, 'scratch')
image_list = os.listdir(image_dir)
blur_list = os.listdir(blur_dir)
scratch_list = os.listdir(scratch_dir)

_data = [1,2,3]
show_text = 0

# record_results('experiment_side.xlsx', _data,'exp1',)


# 근거가 있는지 확인하는 짧은 실험
for idx, (imagename, blurname, scratchname) in enumerate(zip(image_list, blur_list, scratch_list)):
    if idx<3:
        # print(image_path)
        image_path = os.path.join(image_dir, imagename)
        blur_path = os.path.join(blur_dir, blurname)
        scratch_path = os.path.join(scratch_dir, scratchname)

        gray = image_read(image_path)
        gray_blur = image_read(blur_path)
        gray_scratch = image_read(scratch_path)
        image = image_read(image_path)
        image_blur = image_read(blur_path)
        image_scratch = image_read(scratch_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
        # gray_scratch = cv2.cvtColor(image_scratch, cv2.COLOR_BGR2GRAY)

# image를 함수에 입력으로 주고 결과를 받는 부분

# Laplacian part
        fm = variance_of_laplacian(gray)
        fm_blur = variance_of_laplacian(gray_blur)
        fm_scratch = variance_of_laplacian(gray_scratch)

# Sobel part
        # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
        # sobel_blur = cv2.Sobel(gray_blur, cv2.CV_8U, 1, 0, 3)
        # sobel_scratch = cv2.Sobel(gray_scratch, cv2.CV_8U, 1, 0, 3)
        sobel, sobel_fm = sobel_filter(gray)
        sobel_blur, sobel_fm_blur = sobel_filter(gray_blur)
        sobel_scratch, sobel_fm_scratch = sobel_filter(gray_scratch)
        cv2.putText(sobel, "Not blurry: {:.2f}".format(sobel_fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(sobel_blur, "Not blurry: {:.2f}".format(sobel_fm_blur), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(sobel_scratch, "Not blurry: {:.2f}".format(sobel_fm_scratch), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        # laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        # canny = cv2.Canny(image, 100, 255)

        # text = "Not Blurry"
        # cv2.putText(image, "good {}: {:.2f}".format(text, fm), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.putText(image_blur, "blur {}: {:.2f}".format(text, fm_blur), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.putText(image_scratch, "scratch {}: {:.2f}".format(text, fm_scratch), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

# Canny part
        canny = cv2.Canny(gray, 100, 255)
        canny_scratch = cv2.Canny(gray_scratch, 100, 255)
        # cv2.putText(canny, "variance: {:.2f}".format(cv2.Canny(gray, 100, 255).var()), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        # cv2.putText(canny_scratch, "variance: {:.2f}".format(cv2.Canny(gray_scratch, 100, 255).var()), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)


# show image part
        # final_img = cv2.vconcat([image, image_blur])
        # final_img = cv2.vconcat([final_img, image_scratch])
        final_img = cv2.vconcat([image, image_scratch])


# show Sobel part
        # final_img1 = cv2.vconcat([sobel, sobel_blur])
        # final_img1 = cv2.vconcat([final_img1, sobel_scratch])
        final_img1 = cv2.vconcat([sobel, sobel_scratch])


# show Laplacian part
        laplace = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        laplace_scratch = cv2.Laplacian(gray_scratch, cv2.CV_8U, ksize=3)
        final_img2 = cv2.vconcat([laplace, laplace_scratch])


        final_img3 = cv2.vconcat([canny, canny_scratch])



        # final_ = cv2.hconcat([final_img, final_img3])
        # cv2.imshow("Origin and Canny",final_)

        # g = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        # print(f'{final_img.shape}, {final_img3.shape}, {g.shape}')
        sub_imgs = final_img3 - final_img1

        cv2.imshow("Image", final_img)
        cv2.imshow("sobel", final_img1)
        cv2.imshow("laplacian", final_img2)
        cv2.imshow("canny", final_img3)
        cv2.imshow("subtraction", sub_imgs)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # histogram check
        hist_img1 = cv2.calcHist([gray],[0], None, [256],[0,256])
        hist_blur1 = cv2.calcHist([gray_blur], [0], None, [256], [0, 256])
        hist_scratch1 = cv2.calcHist([gray_scratch], [0], None, [256], [0, 256])

        hist_img2 = cv2.calcHist([sobel], [0], None, [256], [0, 256])
        hist_blur2 = cv2.calcHist([sobel_blur], [0], None, [256], [0, 256])
        hist_scratch2 = cv2.calcHist([sobel_scratch], [0], None, [256], [0, 256])

        plt.figure(figsize=(40,40))

        plt.subplot(231),plt.imshow(gray,'gray'),plt.title('good')
        plt.subplot(232), plt.imshow(gray_blur, 'gray'), plt.title('blur')
        plt.subplot(233), plt.imshow(gray_scratch, 'gray'), plt.title('scratch')

        plt.subplot(234), plt.plot(hist_img1, color='r'), plt.plot(hist_blur1, color='g'), plt.plot(hist_scratch1, color='b'), plt.title('histogram of gray')
        plt.xlim([0,256])
        plt.subplot(235), plt.plot(hist_img2, color='r'), plt.plot(hist_blur2, color='g'), plt.plot(hist_scratch2, color='b'), plt.title('histogram of sobel')
        plt.xlim([-10,50])

        # plt.show()


    else:
        break


