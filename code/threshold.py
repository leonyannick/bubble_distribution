import cv2
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os
from scipy.ndimage.morphology import binary_fill_holes
import skimage.morphology, skimage.data
import itertools
import functools
import random
#relative file location
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
print(script_dir)
rel_path = 'pictures/c_P1260783.png'
abs_file_path = os.path.join(script_dir, rel_path)

#save image
save = False
if save is True:
    result_path = 'Beispielbilder/ausschnitt_labor_1_binary.png'
    save_path = os.path.join(script_dir, result_path)


def trackchange(x):
    pass

# create trackbar
cv2.namedWindow("trackbars")

cv2.createTrackbar("median", "trackbars", 5, 50, trackchange)
cv2.createTrackbar("threshold1", "trackbars", 112, 300, trackchange)
cv2.createTrackbar("threshold2", "trackbars", 0, 400, trackchange)
cv2.createTrackbar("aptSize", "trackbars", 3, 10, trackchange)
cv2.createTrackbar("sp", "trackbars", 7, 200, trackchange)
cv2.createTrackbar("sr", "trackbars", 7, 200, trackchange)
cv2.createTrackbar("blockSize", "trackbars", 17, 300, trackchange)
cv2.createTrackbar("constant", "trackbars", 3, 50, trackchange)
cv2.createTrackbar("ksize1", "trackbars", 5, 20, trackchange)
cv2.createTrackbar("ksize2", "trackbars", 0, 20, trackchange)
cv2.createTrackbar("ksize3", "trackbars", 7, 20, trackchange)

cv2.resizeWindow("trackbars", 700, 700)
# opening of sample image
img = cv2.imread(abs_file_path, cv2.IMREAD_GRAYSCALE)
#img = cv2.equalizeHist(img)
#img = cv2.equalizeHist(img)
org = cv2.imread(abs_file_path)
size = 1
img = cv2.resize(img, (0,0), fx=size, fy=size)
org = cv2.resize(org, (0,0), fx=size, fy=size)


while True:
    copy_img = img.copy()
    copy_org = org.copy()
    #cv2.imshow("org", copy)

    #MEDIAN
    md = cv2.getTrackbarPos("median", "trackbars")
    if md % 2 == 0:
        md += 1
    

    #CANNY
    threshold1 = cv2.getTrackbarPos("threshold1", "trackbars")
    threshold2 = cv2.getTrackbarPos("threshold2", "trackbars")
    aptSize = cv2.getTrackbarPos("aptSize", "trackbars")
    if aptSize % 2 == 0:
        aptSize += 1

    #MEAN SHIFT SEGMENTATION
    sp = cv2.getTrackbarPos("sp", "trackbars")
    sr = cv2.getTrackbarPos("sr", "trackbars")
    
    #ADAPTIVE MEDIAN
    blockSize = cv2.getTrackbarPos("blockSize", "trackbars")
    if blockSize % 2 == 0:
        blockSize += 1
    constant = cv2.getTrackbarPos("constant", "trackbars")

    #morphological operations
    ksize1 = cv2.getTrackbarPos("ksize1", "trackbars")
    if ksize1 % 2 == 0:
        ksize1 += 1
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize1,ksize1))
    #kernel = np.ones((5,5), np.uint8)
    ksize2 = cv2.getTrackbarPos("ksize2", "trackbars")
    if ksize2 % 2 == 0:
        ksize2 += 1
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2,ksize2))
    ksize3 = cv2.getTrackbarPos("ksize3", "trackbars")
    if ksize3 % 2 == 0:
        ksize3 += 1
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize3,ksize3))
    
    org_crop = copy_org[702:1287, 87:723]
    cv2.imwrite("graphics/preprocessing/original.png", org_crop)

    img_crop = copy_img[702:1287, 87:723]
    cv2.imwrite("graphics/preprocessing/gray.png", img_crop)

    
    
    median2 = cv2.medianBlur(copy_org, md, 0)
    median = cv2.medianBlur(copy_img, md, 0)
    
    md_crop = median[702:1287, 87:723]
    #cv2.imwrite("graphics/preprocessing/md.png", md_crop)
    #median = cv2.medianBlur(median, md+2, 0)
    #median = cv2.GaussianBlur(copy_img, (md, md), 0)
    #median = cv2.medianBlur(median, md+4, 0)
    #cv2.imwrite("pictures/gaussian_test.png", median)


    #CANNY
    edges = cv2.Canny(median, threshold1, threshold2, apertureSize=aptSize)
    #cv2.imshow("edges", edges)
    #shifted = cv2.pyrMeanShiftFiltering(median2, sp, sr)
    #shifted_2gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, constant)

    threshold_crop = thresh[702:1287, 87:723]
    #cv2.imwrite("graphics/preprocessing/thresh.png", threshold_crop)

    #https://stackoverflow.com/questions/23023805/how-to-get-the-index-and-occurance-of-each-item-using-itertools-groupby
    longst_run = functools.reduce(lambda lst,item: lst + [(item[0], item[1], sum(map(lambda i: i[1], lst)))], [(key, len(list(it))) for (key, it) in itertools.groupby(thresh[0])], [])
    #longest run only 0
    only_zero = []
    for pair in longst_run:
        if pair[0] == 0:
            only_zero.append(pair)
    only_zero = (max(only_zero, key=lambda x:x[1]))

    number = only_zero[1]
    start = only_zero[2]
    zero_pixel_idx = int( start + (number / 2))
    column = zero_pixel_idx
    row = 0

    """
    dilatation = cv2.dilate(thresh, kernel1, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2,ksize2))
    res = binary_fill_holes(dilatation).astype(np.uint8)
    res = np.where(res==1, 255, res)
    erosion = cv2.erode(res, kernel2, iterations=1)
    """
    
    """
    for cnt in contour:
        cv2.drawContours(copy_org, [cnt], 0, 255, -1)
        #cv2.circle(copy_org, tuple(cnt[0][0]), 5, (0,0,255), -1)
        """
    
    
    dilatation = cv2.dilate(thresh, kernel1, iterations=1)
    dilatation_crop = dilatation[702:1287, 87:723]
    cv2.imwrite("graphics/no_dilatation/without_dilatation.png", dilatation_crop)
    """
    # hole filling
    copy1 = dilatation.copy()
    im_floodfill = dilatation.copy()
    h, w = copy1.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    if img[0,0] != 0:
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    imout = copy1 | im_floodfill_inv
    """
    top = 1  # shape[0] = rows
    bottom = top
    left = 1  # shape[1] = cols
    right = left
    
    
    dilatation = cv2.copyMakeBorder(dilatation, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 255)
    
    
    
    dilatation[row,column] = 0
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2,ksize2))
    res = binary_fill_holes(dilatation).astype(np.uint8)
    res = np.where(res==1, 255, res)

    #remove border
    res = np.delete(res, res.shape[1]-1, 1)
    res = np.delete(res, res.shape[0]-1, 0)
    res = np.delete(res, 0, 0)
    res = np.delete(res, 0, 1)

    holefill_crop = res[702:1287, 87:723]
    cv2.imwrite("graphics/no_dilatation/without_dil_holefill.png", holefill_crop)
    """
    labels = skimage.morphology.label(res)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    res[labels != background] = 255
    """
    #https://stackoverflow.com/questions/22310489/filling-holes-in-objects-that-touch-the-border-of-an-image
    erosion = cv2.erode(res, kernel1, iterations=1)

    erosion_crop = erosion[702:1287, 87:723]
    #cv2.imwrite("graphics/preprocessing/erosion.png", erosion_crop)

    #cv2.imwrite("pictures/connected_components_wo_opening_783.png", erosion)
    
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel3)
    opening_crop = opening[702:1287, 87:723]
    #cv2.imwrite("graphics/preprocessing/opening.png", opening_crop)

    #cv2.imwrite("pictures/onnected_components_opening_783.png", opening)
    #size = 0.3
    #copy_img = cv2.resize(copy_img, (0,0), fx=size, fy=size)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(copy_org, contours, -1, (0,255,0), 1)
   
    
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.imshow("image", opening)

    #cv2.imshow("org", copy_org)
    
    if cv2.waitKey(1) == 32:
        break
"""
cv2.imwrite("pictures/test_conc.png", erosion)
cv2.imwrite("pictures/median.png", median)
cv2.imwrite("pictures/canny.png", edges)
cv2.imwrite("pictures/dilatation.png", dilatation)
cv2.imwrite("pictures/erosion.png", erosion)
cv2.imwrite("pictures/hole_fill.png", imout)
"""




cv2.destroyAllWindows()