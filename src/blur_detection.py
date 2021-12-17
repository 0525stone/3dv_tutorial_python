"""
출처 : https://gist.github.com/Pawe1/0e0b2fe4010e4ac6275e1aa1381ddaca

"""
import os
from imutils import paths
import argparse
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# home_dir = os.path.expanduser('~')
# print(home_dir)
#
# print(os.listdir(home_dir))

image_root = '../data/side'# os.path.join(home_dir, )
image_dir = os.path.join(image_root, 'good')
good_dir = os.path.join(image_root, 'scratch')
image_list = os.listdir(image_dir)
good_list = os.listdir(good_dir)


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
