import cv2
import imutils
import numpy as np
import glob
import os
from imutils.object_detection import non_max_suppression

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

test_path = "/home/igor/Downloads/INRIAPerson/Test"


def is_img_ped(file):
    image = cv2.imread(file)
    # image = imutils.resize(image, width=min(400, image.shape[1]))
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return len(pick) > 0


def start_test():
    filenames = glob.iglob(os.path.join(test_path + "/neg", '*'))
    neg_all = 0
    neg_right = 0
    pos_right = 0
    for file in filenames:
        if not is_img_ped(file):
            neg_right = neg_right + 1
        neg_all = neg_all + 1
    filenames = glob.iglob(os.path.join(test_path + "/pos", '*'))
    pos_all = 0
    for file in filenames:
        if is_img_ped(file):
            pos_right = pos_right + 1
        pos_all = pos_all + 1
    print(str.format('all positive : {0}, right : {1}; all neg : {2}, right: {3}', pos_all, pos_right, neg_all,
                     neg_right))


if __name__ == "__main__":
    start_test()
