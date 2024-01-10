import cv2
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os
import itertools
from statistics import mean
from math import pi
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import argparse
from math import pi, sqrt
from scipy.ndimage.morphology import binary_fill_holes
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
    
    erosion = cv2.erode(holefill, kernel, iterations=1)

    #opening
    opening_ksize = thresh_params[4]
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_ksize,opening_ksize))
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, opening_kernel)
    return opening

def angle_2p(a, b): #angle between to vectors in degrees
    dotp = np.dot(a, b)
    alen = np.linalg.norm(a)
    blen = np.linalg.norm(b)
    fraction = dotp / (alen * blen)
    if fraction > 1: #due to float precision fraction sometimes bigger than one (!conflict with arccos)
        fraction = 1
    elif fraction < -1:
        fraction = -1
    angle_rad = np.arccos(fraction)
    angle_deg = angle_rad * (180 / pi)
    return angle_deg

def angles_contour(cnt, cp_params):
    angle_indices = []
    stepsize = 1 #1 = every contour coordinate is used
    distance = cp_params[1]  #distance between start point and middle/end point
    angle_thresh = cp_params[0] #arbitary chosen angle threshold
    size_thresh = 15
    quantity = len(cnt) #amount of contour coordinates
    #discard small bubbles below size_thresh for segmentation, because they mess up the indice logic and they dont matter anyways (aka they dont overlap)
    if quantity > size_thresh:    
        for i in range(0, quantity, stepsize):
            #logic so that indices start from 0 again when they extend quantity
            s = i
            m = i + distance
            e = i + 2 * distance
            if m > quantity:
                m = m % quantity
                e = e % quantity
            elif m == quantity:
                m = m % quantity
                e = e % quantity
            elif e > quantity:
                e = e % quantity
            elif e == quantity:
                e = quantity - 1             
            
            ps = cnt[s][0] #start point
            pm = cnt[m][0] #middle point -> angle gets calculated
            pe = cnt[e][0] #end point
            sm = tuple(map(lambda i, j: i - j, pm, ps)) #vector from start to middle
            em = tuple(map(lambda i, j: i - j, pm, pe)) #vector from end to middle
            angle = angle_2p(sm, em)

            if angle < angle_thresh: #checks if angle is within threshold (angel_thresh)
                if line_inside_contour(cnt, ps, pe) is False:
                    angle_indices.append([angle, m])
    return angle_indices

def line_inside_contour(cnt, ps, pe): #checks if line ps-pe goes through contour -> returns True if line is inside contour
    #calculates midpoint between start and end point
    x_m_point = int((ps[0] + pe[0]) / 2)
    y_m_point = int((ps[1] + pe[1]) / 2)
    m_point = tuple([x_m_point, y_m_point])
    
    #checks if midpoint is inside contour
    inside = cv2.pointPolygonTest(cnt, m_point, measureDist=False) #returns positive value if point inside polygon
    if inside > 0:
        #cv2.circle(original,m_point,2,[0,0,255],-1)
        return True
    return False

def concavepoint_group(idx, contours, concavepoint_indices, locked_concavepoints, cp_params): #redcues group of similiar concavepoints to one
    minDis = cp_params[2] #min Distance between to distinct concavepoints
    if idx == 0: #first point is drawn and added to locked_concavepoints
        first_point_indice = concavepoint_indices[0]
        first_point = tuple(contours[first_point_indice][0])
        locked_concavepoints.append(first_point) 
    else:
        next_point_indice = concavepoint_indices[idx]
        next_point = tuple(contours[next_point_indice][0])
        for points in locked_concavepoints: #next_point is compared to all points in locked_concavepoints, if distance is large to all points it is also added as a new locked_concavepoint, else function returns 0
            eucl = cv2.norm(np.float32(points), np.float32(next_point)) #??? it just works!
            if eucl < minDis:
                return 1
        locked_concavepoints.append(next_point)
    return locked_concavepoints

center = []
ellipse_list = []
def fit_ellipse(image, contour, area_contour, color, draw, add_center):
    corr_factor = 3 #threshold how big fitted ellipses are allowed in comparison to contour area
    global center #bad design, might change later
    global ellipse_list #at this point everything is chaos anyways
    colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),'magenta': (255, 0, 255)}
    if len(contour) > 5:
        arr = np.array(contour) #conversion to numpy array
        ellipse = cv2.fitEllipseDirect(arr)
        area_ellipse = pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2) #NEEDS CONFIRMATION
        if draw:
            if area_ellipse < corr_factor * area_contour:
                cv2.ellipse(image, ellipse, colors["magenta"], 2)
                ellipse_list.append(ellipse)
        if add_center:
            center.append(ellipse[0])

def determine_arches(contour, locked_concavepoints, image, cp_params): #draws ellipses
    mindis_ellipses = cp_params[3]
    area_contour = cv2.contourArea(contour)
    global center
    contour_temp = []
    contour_arches_uneven = []
    contour_arches_even = []
    green = [0,255,0]
    blue = [255,0,0]
    yellow = [0,255,255]
    red = [0,0,255]
    for n in range(len(contour)): #converting contour to the right format for drawing ellipses
        xcor = int(contour[n][0][0])
        ycor = int(contour[n][0][1])
        contour_temp.append((xcor,ycor))
    contour = contour_temp
    if len(locked_concavepoints) % 2 and len(locked_concavepoints) != 1 :
        #UNEVEN number of concave points

        #sort locked_concavepoints
        locked_concavepoints_indice = []
        for point in locked_concavepoints:
            locked_concavepoints_indice.append(contour.index(point))
        locked_concavepoints = [x for _,x in sorted(zip(locked_concavepoints_indice,locked_concavepoints))]

        for idx, point in enumerate(locked_concavepoints):
            idx_start = contour.index(locked_concavepoints[idx])
            
            if idx == len(locked_concavepoints) - 1: #skips last idx in order to connect last element with 0th element again
                idx = -1
            idx_end = contour.index(locked_concavepoints[idx + 1])

            if idx_start > idx_end: #for last iteration remaining points are taken for contour
                idx_start, idx_end = idx_end, idx_start
                contour_arches_uneven.append([item for item in contour if item not in contour[idx_start:idx_end + 1]])
            else: #contourppoints for normal iteration
                contour_arches_uneven.append(contour[idx_start:idx_end + 1])
            
           
    elif len(locked_concavepoints) == 0 or len(locked_concavepoints) == 1:
        #NO CONCAVEPOINTS
        fit_ellipse(image, contour, area_contour, 'magenta', draw=True, add_center=False)
    else:
        #EVEN

        #sort locked_concavepoints
        locked_concavepoints_indice = []
        for point in locked_concavepoints:
            locked_concavepoints_indice.append(contour.index(point))
        locked_concavepoints = [x for _,x in sorted(zip(locked_concavepoints_indice,locked_concavepoints))]

        for idx in range(len(locked_concavepoints)):
            idx1 = contour.index(locked_concavepoints[idx])
            if idx == len(locked_concavepoints) - 1: #skips last idx in order to connect last element with 0th element again
                idx = -1
            idx2 = contour.index(locked_concavepoints[idx + 1])
            if idx1 > idx2: #for last iteration remaining points are taken for contour
                idx1, idx2 = idx2, idx1
                contour_arches_even.append([item for item in contour if item not in contour[idx1:idx2 + 1]])
            else: #contourppoints for normal iteration
                contour_arches_even.append(contour[idx1:idx2 + 1]) 
            #print(f"1:{idx1}")
            #print(f"2:{idx2}")
    
    for contour_arch in contour_arches_uneven: #iterates over contours (UNEVEN concavepoints) and fits + draws ellipse
        fit_ellipse(image, contour_arch, area_contour, 'blue', draw=True, add_center=True)
        
    for contour_arch in contour_arches_even: #iterates over contours (EVEN concavepoints) and fits ellipse WITHOUT drawing
        fit_ellipse(image, contour_arch, area_contour, 'yellow', draw=True, add_center=True)
        
    locked = [] #list of contour indexes of ellipses that are too close together
    for a, b in itertools.combinations(center, 2): #checks all permutations of distances between ellipses
        eucl = cv2.norm(a,b)
        #print(f"eucl:{eucl}")
        if eucl < mindis_ellipses: #number prob needs to be dynamic (in comparison to average distance)
            c = center.index(a)
            d = center.index(b)
            locked.append(c)
            locked.append(d)
            if c < len(contour_arches_even) and d < len(contour_arches_even):
                contour_arches_added = contour_arches_even[c] + contour_arches_even[d]
                fit_ellipse(image, contour_arches_added, area_contour, 'red', draw=True, add_center=False)
    
    for o in range(len(contour_arches_even)): #draws ellipses for EVEN concavepoints
        if o not in locked:
            fit_ellipse(image, contour_arches_even[o], area_contour, 'green', draw=True, add_center=False)
    
    center = []


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
    parser = argparse.ArgumentParser(description='Code for Image Segmentation with Concavepoints and Ellipse Fitting.')
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
    "--cp_params",  # name of the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=int,
    default=[130, 5, 8, 20],  # default if nothing is provided
    help="List of cp parameters: [angle, stepsize, mindis_cp, mindis_ellipses]"
    )
    args = parser.parse_args()

    input_image = args.input_image
    original, image = read_image(input_image)
    thresh_params = args.thresh_params
    cp_params = args.cp_params

    thresh = threshold(image, thresh_params)
    
    


    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(original, contours, -1, (0,255,0), 1)

    for idx, contour in enumerate(contours): #goes through all different detected contours
        concavepoint_indices = []
        result = angles_contour(contour, cp_params)
        result = (sorted(result, reverse=False))
        if not not result: #skip empty lists
            for item in result: #goes through result list and picks all points with an angle below certain value
                concavepoint_indices.append(item[1])

        #draw concavepoints
        locked_concavepoints = [] #list of confirmed concavepoints -> for concavepoint_group function
        for idx, indice in enumerate(concavepoint_indices):
            concavepoint_group(idx, contour, concavepoint_indices, locked_concavepoints, cp_params)

        for points in locked_concavepoints:
            cv2.circle(original,points,3,[0,0,255],-1)

        determine_arches(contour, locked_concavepoints, original, cp_params) #determines perimeter arches from concavepoints -> and calls fit_ellipses function

    

    base_name = os.path.basename(input_image)
    name = os.path.splitext(base_name)[0]

    #save data
    output = args.output
    output_csv = output + name + "_cp.csv"
    data_output(ellipse_list, output_csv) 

    #save image
    output_image = output + name + "_cp.png"
    cv2.imwrite(output_image, original)

    #save threshold image
    output_thresh = output + name + "_thresh_cp.png"
    cv2.imwrite(output_thresh, thresh)

if __name__ == "__main__":
    main()    

#polygon approx. -> reduction of contour points
""" 
epsilon = 0.01 * cv2.arcLength(n, True)
approx = cv2.approxPolyDP(n, epsilon, True)
cv2.drawContours(original, [approx], 0, (255,0,0), 1)
print(f"poly:{len(approx)}")
print(f"cnt:{len(n)}")
print("\n")
"""