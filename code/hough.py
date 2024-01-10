import cv2
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os
from math import pi, sqrt
import argparse
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import functools
import itertools

def read_image(input_image):
    original = cv2.imread(input_image)
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    return original, image

def threshold(image, thresh_params):
    #median blur
    median_ksize = thresh_params[0]
    if median_ksize == 0:
        median = image
    else:
        median = cv2.medianBlur(image, median_ksize, 0)
        #median = cv2.GaussianBlur(image, (median_ksize, median_ksize), 0)

    #adaptive threshold
    blockSize = thresh_params[1]
    constant = thresh_params[2]
    thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, constant)

    #find longest consecutive sequence of same integer
    longst_run = functools.reduce(lambda lst,item: lst + [(item[0], item[1], sum(map(lambda i: i[1], lst)))], [(key, len(list(it))) for (key, it) in itertools.groupby(thresh[0])], [])
    #longest run of only 0
    only_zero = []
    for pair in longst_run:
        if pair[0] == 0:
            only_zero.append(pair)
    only_zero = (max(only_zero, key=lambda x:x[1]))

    #index of middle if longst sequence
    number = only_zero[1]
    start = only_zero[2]
    zero_pixel_idx = int( start + (number / 2))
    column = zero_pixel_idx
    
    #morphological operations
    morph_ksize = thresh_params[3]
    if morph_ksize == 0:
        dilatation = thresh
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize,morph_ksize))
        #kernel = np.ones((5,5), np.uint8)
        dilatation = cv2.dilate(thresh, kernel, iterations=1)

    #1 pixel white border around image
    top = 1  # shape[0] = rows
    bottom = top
    left = 1  # shape[1] = cols
    right = left
    dilatation = cv2.copyMakeBorder(dilatation, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 255)
    
    #1 pixel gap in top border
    dilatation[0,column] = 0
    
    # hole filling
    holefill = binary_fill_holes(dilatation).astype(np.uint8)
    holefill = np.where(holefill==1, 255, holefill)

    #remove border
    holefill = np.delete(holefill, holefill.shape[1]-1, 1)
    holefill = np.delete(holefill, holefill.shape[0]-1, 0)
    holefill = np.delete(holefill, 0, 0)
    holefill = np.delete(holefill, 0, 1)
    
    if morph_ksize == 0:
        erosion = holefill
    else:
        erosion = cv2.erode(holefill, kernel, iterations=1)

    #opening
    opening_ksize = thresh_params[4]
    if opening_ksize == 0:
        opening = erosion
    else:
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_ksize,opening_ksize))
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, opening_kernel)
    return opening

def hough_circle(thresh, image, hough_parameters):
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1 ,minDist=hough_parameters[0], param1=1, param2=hough_parameters[1], minRadius=0, maxRadius=100)
    if circles is None: #if no circles are found
        return circles
    else:
        circles = np.uint16(np.around(circles))
    for idx,i in enumerate(circles[0,:]):
        # draw the outer circle
        cv2.circle(image,(i[0],i[1]),i[2],(255,0,255),2)
        # draw the center of the circle
        cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
        


    return circles[0]
    

def data_output(circle_list, output_data):
    #RESULTS DICTIONARY
    results = {}
    area = []
    perimeter = []
    center = []
    count = []
    a_semimajor = []
    b_semiminor = []
    eccentricity = []
    ellipses_list = []
    angle_ver = []
    d_eq = []
    #assigning results
    if circle_list is not None:
        for idx, circle in enumerate(circle_list):
            #area.append(cv2.contourArea(i))
            c = (circle[0],circle[1])
            a_semimajor.append(circle[2])
            b_semiminor.append(circle[2])
            r = circle[2]
            a = pi * r * r
            u = 2 * pi * r
            d_eq.append(2 * r)
            area.append(a)
            perimeter.append(u)
            center.append(c)
            count.append(idx)
            eccentricity.append(0)
            angle_ver.append(0)
            ellipses_list.append((c, (2*r, 2*r), 0))

    results["count"] = count
    results["area"] = area
    results["perimeter"] = perimeter
    results["center"] = center
    results["d_eq"] = d_eq
    results["a_semimajor"] = a_semimajor
    results["b_semiminor"] = b_semiminor
    results["angle_ver"] = angle_ver
    results["eccentricity"] = eccentricity
    results["ellipses_list"] = ellipses_list

    # Create the Pandas DataFrame
    df = pd.DataFrame(results)
    
    # Export the dataframe to a csv file
    df.to_csv(path_or_buf = output_data, index = None, header=True)
    return df

def main():
    #command-line input
    parser = argparse.ArgumentParser(description='Code for Image Segmentation with Hough Algorithm.')
    parser.add_argument('input_image', help='Path to input image.', type=str)
    parser.add_argument('output', help='Path to output data.', type=str)
    parser.add_argument(
    "--thresh_params",  # name of the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=int,
    default=[5, 17, 3, 5, 7],  # default if nothing is provided
    help="List of threshold parameters: [mediumk., blocksz, const., dil/erok., openingk.]"
    )
    parser.add_argument(
    "--hough_params",  # name of the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=int,
    default=[32, 13],  # default if nothing is provided
    help="List of hough parameters: [minDis, param2]"
    )
    args = parser.parse_args()

    thresh_params = args.thresh_params
    hough_params = args.hough_params
    input_image = args.input_image
    original, image = read_image(input_image)
    thresh = threshold(image, thresh_params)

    circle_list = hough_circle(thresh, original, hough_params)

    base_name = os.path.basename(input_image)
    name = os.path.splitext(base_name)[0]

    #save data
    output = args.output
    output_csv = output + name + "_h.csv"
    data_output(circle_list, output_csv) 

    #save image
    output_image = output + name + "_h.png"
    #cv2.imwrite(output_image, original)

if __name__ == "__main__":
    main()    








# canny edge detection
#laplacian = cv2.Laplacian(median, cv2.CV_64F)
#laplacian1 = laplacian/laplacian.max()
#edges1 = cv2.Canny(median, 50, 200, 3)
# edges2 = cv2.Canny(median, 100, 300, 3)
# edges2copy = cv2.Canny(median, 100, 300, 3)

#ret, thresh = cv2.threshold(laplacian1, 0, 255, cv2.THRESH_BINARY)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(img, contours, -1, (0,0,0), 2)
##cv2.imshow("testsf", thresh)
# hough transform

