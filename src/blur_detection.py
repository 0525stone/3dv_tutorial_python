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

home_dir = os.path.expanduser('~')
print(home_dir)

print(os.listdir(home_dir))

image_root = '../data/light_distract'# os.path.join(home_dir, )
image_dir = os.path.join(image_root, 'blur')
image_list = os.listdir(image_dir)
# good_list = os.listdir()

for idx, image_path in enumerate(image_list):
    if idx<10:
        print(image_path)
        image_path = os.path.join(image_dir, image_path)

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"

        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)







    else:
        break
