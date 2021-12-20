"""
출처 : https://gist.github.com/Pawe1/0e0b2fe4010e4ac6275e1aa1381ddaca

실험 기록 위하여 엑셀 시트 접근, 데이터 기록 파트 추가
"""
import os
from imutils import paths
from pandas import DataFrame
import pandas as pd

import argparse
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sobel_filter():
    pass


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

        for idx,img_path in enumerate(img_list,1):
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

# record_results('experiment_side.xlsx', _data,'exp1',)

# 근거가 있는지 확인하는 짧은 실험
for idx, (imagename, blurname, scratchname) in enumerate(zip(image_list, blur_list, scratch_list)):
    if idx<20:
        # print(image_path)
        image_path = os.path.join(image_dir, imagename)
        blur_path = os.path.join(blur_dir, blurname)
        scratch_path = os.path.join(scratch_dir, scratchname)

        image = cv2.imread(image_path)
        image_blur = cv2.imread(blur_path)
        image_scratch = cv2.imread(scratch_path)
        image = cv2.pyrDown(image)
        image_blur = cv2.pyrDown(image_blur)
        image_scratch = cv2.pyrDown(image_scratch)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
        gray_scratch = cv2.cvtColor(image_scratch, cv2.COLOR_BGR2GRAY)

        fm = variance_of_laplacian(gray)
        fm_blur = variance_of_laplacian(gray_blur)
        fm_scratch = variance_of_laplacian(gray_scratch)

        text = "Not Blurry"

        cv2.putText(image, "good {}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.putText(image_blur, "blur {}: {:.2f}".format(text, fm_blur), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.putText(image_scratch, "scratch {}: {:.2f}".format(text, fm_scratch), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        final_img = cv2.vconcat([image, image_blur])
        final_img = cv2.vconcat([final_img, image_scratch])
        cv2.imshow("Image", final_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


    else:
        break


