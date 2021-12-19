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


def record_results():
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

    writer = pd.ExcelWriter('experiment_side.xlsx')

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
            # print(f'{type(img_)},  {img_}')
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
good_dir = os.path.join(image_root, 'scratch')
image_list = os.listdir(image_dir)
good_list = os.listdir(good_dir)


# record_results()

# 근거가 있는지 확인하는 짧은 실험
for idx, (imagename, goodname) in enumerate(zip(image_list, good_list)):
    if idx<20:
        # print(image_path)
        image_path = os.path.join(image_dir, imagename)
        good_path = os.path.join(good_dir, goodname)

        image = cv2.imread(image_path)
        image_good = cv2.imread(good_path)
        image = cv2.pyrDown(image)
        # image = cv2.pyrDown(image)
        # image_good = cv2.pyrDown(image_good)
        image_good = cv2.pyrDown(image_good)
        # image = cv2.pyrDown(image)
        # image_good = cv2.pyrDown(image_good)

        # gray, gray_good = image, image_good

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_good = cv2.cvtColor(image_good, cv2.COLOR_BGR2GRAY)
        fm_good = variance_of_laplacian(gray_good)
        fm = variance_of_laplacian(gray)

        text = "Not Blurry"

        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.putText(image_good, "{}: {:.2f}".format(text, fm_good), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        final_img = cv2.vconcat([image, image_good])
        cv2.imshow("Image", final_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


    else:
        break


