"""
2021.12.17
from image directory make name list
"""
import os

def extract_name_dir(dir_name, savename):

    f = open(savename, 'w')
    img_list = os.listdir(dir_name)
    result = []
    print(img_list)
    img_list = sorted(img_list)
    for name in img_list:
        # print(name)
        result.append(name)
        temp = name+'\n'
        f.write(temp)
    print(f'length of image list : {len(result)}')
    f.close()



_root = '../data/side'
_clss = ['blur','scratch','good']
_cls = 'blur'
_file_dir = os.path.join(_root,_cls)
_save = 'img_'+_cls+'.txt'

# extract_name_dir(_file_dir, _save)

for c in _clss:
    _file_dir = os.path.join(_root, c)
    _save = 'img_' + c + '.txt'
    extract_name_dir(_file_dir, _save)
    # pass

