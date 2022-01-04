"""
EDA

When I first got Dataset, I want to know...
- how many data are in the directory
- how many folder are in dataset

Image dataset
- Distribution of image size
- What image format is
-
"""
import glob
import os.path
import json
import random
import cv2
import re
import pickle



def img_list_from_json(json_name):
    """
    random sampling 하여 데이터를 직접 확인하기 위한 코드

    _electric_append.json 에서는 key 값이 데이터 이름

    """
    # reading json file as dictionary
      # top directory name of filename should be changed from mnt to deephigh blah blah
    img_list = []
    with open(json_name) as json_file:
        data = json.load(json_file)

        # get name from dictionary and make list(may save by .txt file later)
        for idx, name in enumerate(data.keys()):
            img_list.append(name)

        print(f'image list : {len(img_list)}')

    return img_list


def random_sample_check(label_file,n_samples):
    with open(label_file,'r') as f:
        readList = f.readlines()
    random.seed(25)
    r_sample = random.sample(readList, n_samples)
    print(f'random sample {len(r_sample)}')
    return r_sample


def check_image(img_dir):
    home_dir = os.path.expanduser('~')
    img_temp = img_dir.split('/8TB/')[1]
    img_name = img_temp.split('\n')[0]
    print(img_name)
    image = cv2.imread(img_name)#,cv2.IMREAD_GRAYSCALE)

    cv2.imshow("check", image)
    # record if it is noise or not by  waitkey
    # cv2.waitKey() # reference
    key = cv2.waitKey()
    if key == ord('f'):
        cv2.destroyAllWindows()
        return True, img_name
    else:
        cv2.destroyAllWindows()
        return False, img_name


# ################################################################################################
# Check image list
home_dir = os.path.expanduser('~') # home directory check

file_dir = "./git/numberplate/data/detector/*_electric_append.json"
file_dir = os.path.join(home_dir, file_dir)
file_list1 = glob.glob(file_dir)
file_list1.sort()

img_list = []

print(home_dir)
print(file_dir)

r_sample = random_sample_check('train_set_ver1.txt',10)

noise_list = []
for img_ in r_sample:
    _flag, img_name = check_image(img_)
    if (_flag):
        noise_list.append(img_name)
print(f'noise picture : {len(noise_list)}')


# ################################################################################################
# # making image list from json
# home_dir = os.path.expanduser('~') # home directory check
#
# file_dir = "./git/numberplate/data/detector/*_electric_append.json"
# file_dir = os.path.join(home_dir, file_dir)
# file_list1 = glob.glob(file_dir)
# file_list1.sort()
#
# img_list = []
#
# # first list
# for idx, file_name in enumerate(file_list1):
#     temp_img_list = img_list_from_json(file_name)
#
#     img_list.extend(temp_img_list)
#     print(f'after read json file : {len(img_list)}')
#
# # second list
# # file_list2 = "./git/numberplate/data/detector/coco_numberplate_to_text_pannel_convert.json"
# # file_list2 = os.path.join(home_dir, file_list2)
# # # temp_img_list = img_list_from_json(file_list2)
# # img_list.extend(temp_img_list)
#
# print(f'after read second json file : {len(img_list)}')
#
# # save to .txt file
# savename = 'train_set_ver1.txt'
# with open(savename, 'w+') as f:
#     f.write('\n'.join(img_list))


# # save by pickle
# with open(savename, 'wb') as f:
#     pickle.dump(img_list,f)




