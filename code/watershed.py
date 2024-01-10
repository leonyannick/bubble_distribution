import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import os
import imutils
import argparse
import pandas as pd
from math import pi, sqrt
from scipy.ndimage.morphology import binary_fill_holes
import itertools
import functools

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

def watershed_seg(image, ws_params):   
    minDis = ws_params[0]
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, min_distance=minDis, labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    ellipse_list = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background', so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255
        #cv2.imshow("mask", mask)
        #cv2.waitKey(0)
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        contour = [c]
        #cv2.drawContours(image, contour, -1, (0,0,255), 5)

        for cnt in contour:
            #area_contour = cv2.contourArea(cnt)
            #corr_factor = 1.2
            #size_minimum = 0.3 * area_contour
            if len(cnt) > 5:
                ellipse = cv2.fitEllipseDirect(cnt)
                #area_ellipse = pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
                #if area_ellipse < corr_factor * area_contour and area_ellipse > size_minimum:
                ellipse_list.append(ellipse)
     

    #for ellipse in ellipse_list:
                #cv2.ellipse(labels, ellipse, (200, 0, 200), 2)

    return distance, labels, ellipse_list

def plot_segmentation(image, distance, labels):
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    #plt.show()

    return labels

def data_output(ellipse_list, output_data):
    #RESULTS DICTIONARY
    results = {}
    count, area, perimeter, center, d_eq, a_semimajor, b_semiminor, angle_ver, eccentricity, ellipses_list = ([] for i in range(10))
    
    for counter, ellipse in enumerate(ellipse_list):
        #center
        center_x = ellipse[0][0]
        center_y = ellipse[0][1]
        c = (center_x, center_y)
        center.append(c)
        #a_semimajor/b_semiminor
        a = (ellipse[1][1] / 2)
        b = (ellipse[1][0] / 2)
        a_semimajor.append(a)
        b_semiminor.append(b)
        #perimeter
        h = (a-b) ** 2 / (a+b) ** 2
        peri_ramanujan = pi * (a + b) * (1 + 3 * h / (10 + sqrt(4 - 3 * h))) #ramanujan perimeter approx.
        perimeter.append(peri_ramanujan)
        #count 
        count.append(counter)
        #area
        area_ellipse = pi * a  * b
        area.append(area_ellipse)
        #equivalent diameter
        d = 2 * sqrt(area_ellipse / pi)
        d_eq.append(d)
        #angle_ver
        angle = ellipse[2]
        angle_ver.append(angle) 
        #eccentricity
        e = sqrt(1 - (b ** 2 / a ** 2))
        eccentricity.append(e)
        #ellipse_list
        ellipses_list.append(ellipse)

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
    parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance Transform and Watershed Algorithm.')
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
    "--ws_params",  # name of the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=int,
    default=[0],  # default if nothing is provided
    help="List of watershed parameters: [minDis]"
    )
    args = parser.parse_args()

    thresh_params = args.thresh_params
    ws_params = args.ws_params
    input_image = args.input_image

    original, image = read_image(input_image)
    thresh_new = threshold(image, thresh_params)

    distance, labels, ellipse_list = watershed_seg(thresh_new, ws_params)
    proc_image = plot_segmentation(original, distance, labels)

    base_name = os.path.basename(input_image)
    name = os.path.splitext(base_name)[0]

    #save data
    output = args.output
    output_csv = output + name + "_ws.csv"
    data_output(ellipse_list, output_csv) 

    #save image
    for ellipse in ellipse_list:
                cv2.ellipse(original, ellipse, (200, 0, 200), 2)
    output_image = output + name + "_ws.png"
    cv2.imwrite(output_image, original)

    output_label = output + name + "_label_ws.png"
    

    print(labels.shape)
    fig = plt.figure(frameon=False, figsize=(2.000, 1.650), dpi=1000)
    #fig.set_size_inches(w,h)


    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


    ax.imshow(labels, cmap=plt.cm.nipy_spectral)




    fig.savefig(output_label)
    #plt.show()

    
    
    



if __name__ == "__main__":
    main()