"""
출처 : https://gist.github.com/Pawe1/0e0b2fe4010e4ac6275e1aa1381ddaca

실험 기록 위하여 엑셀 시트 접근, 데이터 기록 파트 추가
"""
from imutils import paths
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot as plt
from itertools import cycle

import os
import cv2
import random
import argparse


def temp_dict_fill_data(data_dir):
    data_dict = {}
    image_dict = {}

    name_list = []
    lap_list = []
    canny_list = []
    sobel_list = []

    image_list = os.listdir(data_dir)
    for idx, imagename in enumerate(image_list):
        image_path = os.path.join(data_dir, imagename)
        gray = image_read(image_path, gray_flag)
        image = image_read(image_path, 0)

    # result of image process
        laplacian, fm = laplacian_filter(gray, show_text)
        sobel, sobel_fm = sobel_filter(gray, show_text)
        canny, canny_fm = canny_filter(gray, show_text)

        name_list.append(imagename)
        lap_list.append(fm)
        canny_list.append(canny_fm)
        sobel_list.append(sobel_fm)

        image_dict[idx] = {'name':imagename,'laplacian':laplacian, 'sobel':sobel, 'canny':canny}
    data_dict['name'] = name_list
    data_dict['Laplacian'] = lap_list
    data_dict['Canny'] = canny_list
    data_dict['Sobel'] = sobel_list
    return data_dict, image_dict


def temp_dict_to_df(dict_data):
    new_dict = {'good name': dict_data['good name'], 'good score': dict_data['good score'],
                 'scratch name': dict_data['scratch name'],
                 'scratch score': dict_data['scratch score'], 'blur name': dict_data['blur name'],
                 'blur score': dict_data['blur score']}
    new_df = DataFrame(new_dict)

    return new_df


def image_read(image_path, gray_flag):
    if(gray_flag):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.pyrDown(image,cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.pyrDown(image)

    if(len(image.shape)==2):
        h,w = image.shape
    elif(len(image.shape)==3):
        h,w,c = image.shape
    else:
        print('dimension of image is over 3, need to check.')

    return image[:,int(w/3):int(2*w/3)]


def variance_of_laplacian(image, flag):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def histogram_check(gray, gray_blur, gray_scratch, sobel,sobel_blur, sobel_scratch):
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

    plt.show()


def canny_filter(gray_img, flag):
    canny = cv2.Canny(gray_img, 100, 255)
    fm = cv2.Canny(gray_img, 100, 255).var()
    if(flag):
        cv2.putText(canny, "Blurry: {:.2f}".format(fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    return canny, fm


def laplacian_filter(gray_img, flag):
    # print(f'input shape : {gray_img.shape}')
    lap_img = cv2.Laplacian(gray_img, cv2.CV_8U, ksize=3)
    fm = cv2.Laplacian(gray_img, cv2.CV_8U, ksize=3).var()
    if(flag):
        cv2.putText(lap_img, "Blurry: {:.2f}".format(fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    return lap_img, fm


def sobel_filter(gray_img, flag=0):
    sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3)
    # fm = variance_of_laplacian(sobel)
    fm = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, 3).var()
    h, w = gray_img.shape
    if(flag):
        cv2.putText(sobel, "Blurry: {:.2f}".format(fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
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
gray_flag = 1

# record_results('experiment_side.xlsx', _data,'exp1',)

# dictionary to save scores
lap_dict = {'good':{},'blur':{},'scratch':{}}
canny_dict = {'good':{},'blur':{},'scratch':{}}
sobel_dict = {'good':{},'blur':{},'scratch':{}}

# try image processing on entire image data
good_dict, good_img_dict = temp_dict_fill_data(image_dir)
scratch_dict, scratch_img_dict = temp_dict_fill_data(scratch_dir)
blur_dict, blur_img_dict = temp_dict_fill_data(blur_dir)
# print(f'{list(good_dict.keys())}')


# save in excel file
writer = pd.ExcelWriter('experiment_.xlsx')

good_df = DataFrame(good_dict)
scratch_df = DataFrame(scratch_dict)
blur_df = DataFrame(blur_dict)

good_df.to_excel(writer, sheet_name='good')
scratch_df.to_excel(writer, sheet_name='scratch')
blur_df.to_excel(writer, sheet_name='blur')

writer.save()













# quick and short experiment to validate proposed process
# 근거가 있는지 확인하는 짧은 실험
# for idx, (imagename, blurname, scratchname) in enumerate(zip(image_list, blur_list, scratch_list)):
#     # if idx<70:
# # #####################################################################################################################
#     print(f'{idx+1} th image, picture :')
#     # print(image_path)
#     image_path = os.path.join(image_dir, imagename)
#     blur_path = os.path.join(blur_dir, blurname)
#     scratch_path = os.path.join(scratch_dir, scratchname)
#
#     gray = image_read(image_path, gray_flag)
#     gray_blur = image_read(blur_path, gray_flag)
#     gray_scratch = image_read(scratch_path, gray_flag)
#
#     image = image_read(image_path, 0)
#     image_blur = image_read(blur_path, 0)
#     image_scratch = image_read(scratch_path, 0)
#
#
# # Processing part
# # Laplacian part
#     laplacian, fm = laplacian_filter(gray, show_text)
#     laplacian_blur, fm_blur = laplacian_filter(gray_blur, show_text)
#     laplacian_scratch, fm_scratch = laplacian_filter(gray_scratch, show_text)
#     lap_dict['good'][imagename] = fm
#     lap_dict['scratch'][scratchname] = fm_scratch
#     lap_dict['blur'][blurname] = fm_blur
#
# # Sobel part
#     sobel, sobel_fm = sobel_filter(gray, show_text)
#     sobel_blur, sobel_fm_blur = sobel_filter(gray_blur, show_text)
#     sobel_scratch, sobel_fm_scratch = sobel_filter(gray_scratch, show_text)
#     sobel_dict['good'][imagename] = sobel_fm
#     sobel_dict['scratch'][scratchname] = sobel_fm_scratch
#     sobel_dict['blur'][blurname] = sobel_fm_blur
#
# # Canny part
#     canny, canny_fm = canny_filter(gray, show_text)
#     canny_blur, canny_fm_blur = canny_filter(gray_blur, show_text)
#     canny_scratch,canny_fm_scratch = canny_filter(gray_scratch, show_text)
#     canny_dict['good'][imagename] = canny_fm
#     canny_dict['scratch'][scratchname] = canny_fm_scratch
#     canny_dict['blur'][blurname] = canny_fm_blur
#
# # subtract part
#     # empty yet
#


# # concatenate to show
#     final_img = cv2.vconcat([image, image_scratch])
#     final_img1 = cv2.vconcat([sobel, sobel_scratch])
#     final_img2 = cv2.vconcat([laplacian, laplacian_scratch])
#     final_img3 = cv2.vconcat([canny, canny_scratch])
#
# final showing output part
    # cv2.imshow("Image", final_img)
    # cv2.imshow("sobel", final_img1)
    # cv2.imshow("laplacian", final_img2)
    # cv2.imshow("canny", final_img3)
    # # cv2.imshow("subtraction", sub_imgs)
    # cv2.waitKey()
    # cv2.destroyAllWindows()



# #####################################################################################################################
#     else:
#         break


# # # histogram check
# # histogram_check(gray, gray_blur, gray_scratch, sobel,sobel_blur, sobel_scratch)
#
# # save in excel file
# writer = pd.ExcelWriter('experiment_.xlsx')
#
# sobel_df = temp_dict_to_df(sobel_dict)
# lap_df = temp_dict_to_df(lap_dict)
# canny_df = temp_dict_to_df(canny_dict)
#
# canny_df.to_excel(writer, sheet_name='canny')
# sobel_df.to_excel(writer, sheet_name='sobel')
# lap_df.to_excel(writer, sheet_name='laplacian')
#
# writer.save()