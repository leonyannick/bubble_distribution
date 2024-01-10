import cv2
import numpy as np
import matplotlib.pyplot as plt 
import random
import cv2
import itertools
from collections import Counter
import argparse
from math import pi, sqrt
import pandas as pd
import csv

def randomellipse(height, width, a_semimajor, b_semiminor, angle):
    a_deviation = a_semimajor * 0.25
    b_deviation = b_semiminor * 0.25
    majoraxis = abs(int(random.normalvariate(a_semimajor, a_deviation)))
    minoraxis = abs(int(random.normalvariate(b_semiminor, b_deviation)))
    if majoraxis < minoraxis:
        majoraxis, minoraxis = minoraxis, majoraxis
    color = (80,80,80)
    radius = (majoraxis + minoraxis) / 2
    thickness = int(1 * 0.15 * radius)

    center = [None] * 2
    center[0] = int(random.uniform(0, width))
    center[1] = int(random.uniform(0, height))


    ellipse = (tuple(center), (minoraxis, majoraxis), angle)
    ellipse_data = (tuple(center), (minoraxis + thickness, majoraxis + thickness), angle)
    return ellipse, ellipse_data, color, thickness

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
    parser = argparse.ArgumentParser(description='Code for generating artificial ellipses.')
    parser.add_argument('output_image', help='Path to output image.', type=str)
    parser.add_argument('output_data', help='Name and path for output data.', type=str)
    args = parser.parse_args()
    output_image = args.output_image
    output_data = args.output_data

    #input parameters
    height = 1650
    width = 2000      
    countapprox = 100

    #create gray background
    image = np.zeros((height, width, 3), np.uint8)
    image[0:height,0:width] = 200

    #count random normal distribution
    count = int(random.normalvariate(countapprox, 20))
    #count = 10

    #axes,angle data from real images 
    data = "evaluation/ellipse_generator_calibration_2/data.csv"
    axes = []
    angle = []
    with open(data, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            axes.append(eval(row["axes"]))
            angle.append(eval(row["angle"]))
    

    #loops through sample output and generates random ellipses which are saved in ellipses_list
    achsen = random.choices(axes, k=count)
    winkel = random.choices(angle, k=count)
    ellipses_list = [] 
    ellipses_data_list = [] #list of ellipses from data
    thick = [] 
    for a, w in zip(achsen, winkel):
        a_semimajor = a[0]
        b_semiminor = a[1]
        winkel = w
        ellipse, ellipse_data, color, thickness = randomellipse(height, width, a_semimajor, b_semiminor, winkel)
        ellipses_list.append(ellipse)
        ellipses_data_list.append(ellipse_data)
        thick.append(thickness)
    
    #ellipses are drawn on image (gray background)
    for ellipse, thickness in zip(ellipses_list,thick):
        cv2.ellipse(image, ellipse, color, thickness)


    #intensity of bubble borders
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)
    mask = np.zeros(image.shape, np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255,255,255),5)
    output = np.where(mask==np.array([255, 255, 255]), blurred_img, image)
    
    #save data
    data_output(ellipses_data_list, output_data)

    #save image
    cv2.imwrite(output_image, output)
    """
    #show image
    cv2.namedWindow("blurred", cv2.WINDOW_NORMAL)
    cv2.imshow("blurred", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    """

if __name__ == "__main__":
    main()