import cv2
import numpy as np
from sklearn.metrics import jaccard_score
from scipy.stats import gaussian_kde
import csv
from statistics import mean
from skimage.io import imread
from math import pi, sqrt
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from shapely.geometry.point import Point
from shapely import affinity
import glob
import os
from itertools import chain
import matplotlib as mpl

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    https://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    if not ellr.is_valid:
        return 0
    return ellr

def groundtruth(groundtruth_csv):
    groundtruth_ellipses = []
    with open(groundtruth_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            groundtruth_ellipses.append(eval(row["ellipses_list"]))
    return groundtruth_ellipses

def seg_output(seg_output_csv):
    seg_output_ellipses = []
    with open(seg_output_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seg_output_ellipses.append(eval(row["ellipses_list"]))
    return seg_output_ellipses

def jaccard_f1(groundtruth_ellipses, seg_output_ellipses):
    tp = 0 #true positives -> correct detection of existing object
    fp = 0 #false postitives -> detection of an object that does not exist
    fn = 0 #false negatives -> failing to detect an existing object

    #functionaliy: 2 for loops to go through groundtruth/segmented ellipses -> draw  individual ellipses and calculate jaccard index (image array need to be flat 1D)
    #  -> specify certain threshold for correct segmentation -> assign output to tp, fp, fn -> calculate recall, precision and f1-score
    average_jaccard = []
    for ellipse_g in groundtruth_ellipses:
        jaccard_score_list = []
        ellipse_g = create_ellipse(ellipse_g[0],ellipse_g[1],ellipse_g[2])
        area_g = ellipse_g.area
        for idx, ellipse_seg in enumerate(seg_output_ellipses):

            ellipse_seg = create_ellipse(ellipse_seg[0], ellipse_seg[1], ellipse_seg[2])
            if ellipse_seg == 0: #cp algorithm returns empty polygons 
                continue
            area_seg = ellipse_seg.area
            intersect = ellipse_g.intersection(ellipse_seg)
            area_intersection = intersect.area
            area_union = area_g + area_seg - area_intersection
            score = area_intersection / area_union
            jaccard_score_list.append(score)
        
        jaccard_score_list.sort(reverse=True) 
        if len(jaccard_score_list) == 0: #in case there are no detections
            jaccard_score_list = [0]

        if jaccard_score_list[0] > 0.5:
            tp += 1
        else:
            fn += 1

        average_jaccard.append(jaccard_score_list[0])        
            
    average_jaccard = mean(average_jaccard) #-> average jaccard index (refers to groundtruth elements)
    
    
    for ellipse_seg in seg_output_ellipses:
        jaccard_score_list = []
        ellipse_seg = create_ellipse(ellipse_seg[0],ellipse_seg[1],ellipse_seg[2])
        if ellipse_seg == 0:
            print("segment mask too small to fit ellipse")
            jaccard_score_list.append(0)
            fp += 1
            continue
        area_seg = ellipse_seg.area
        for ellipse_g in groundtruth_ellipses:
            ellipse_g = create_ellipse(ellipse_g[0], ellipse_g[1], ellipse_g[2])
            area_g = ellipse_g.area
            intersect = ellipse_seg.intersection(ellipse_g)
            area_intersection = intersect.area
            area_union = area_g + area_seg - area_intersection
            score = area_intersection / area_union
            jaccard_score_list.append(score)
            
        jaccard_score_list.sort(reverse=True) 

        if jaccard_score_list[0] < 0.5:
            fp += 1
    try:    #in case there are no detections -> precision = 0
        precision = tp / (tp + fp)
    except:
        precision = 0
    recall = tp / (tp + fn)
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall / (precision + recall))

    return [tp, fp, fn, precision, recall, f1_score, average_jaccard]

def sauter(groundtruth_ellipses, seg_output_ellipses):
    d_eq_list = []
    for ellipse_seg in seg_output_ellipses:
        a_semimajor = ellipse_seg[1][1]
        b_semiminor = ellipse_seg[1][0]
        area = pi * a_semimajor * b_semiminor
        d_eq = 2 * sqrt(area / pi)
        d_eq_list.append(d_eq)
    if len(d_eq_list) == 0: #in case list is empty
        sauter_seg = 0
    else:    
        sauter_seg = sum([d_eq ** 3 for d_eq in d_eq_list]) / sum([d_eq ** 2 for d_eq in d_eq_list])
    
    d_eq_list = []
    for ellipse_g in groundtruth_ellipses:
        a_semimajor = ellipse_g[1][1]
        b_semiminor = ellipse_g[1][0]
        area = pi * a_semimajor * b_semiminor
        d_eq = 2 * sqrt(area / pi)
        d_eq_list.append(d_eq)
    sauter_g = sum([d_eq ** 3 for d_eq in d_eq_list]) / sum([d_eq ** 2 for d_eq in d_eq_list])

    return [sauter_seg, sauter_g]

def probability_density(groundtruth_ellipses, seg_output_ellipses):
    d_eq_list_seg = []
    for ellipse_seg in seg_output_ellipses:
        a_semimajor = ellipse_seg[1][1]
        b_semiminor = ellipse_seg[1][0]
        area = pi * a_semimajor * b_semiminor
        d_eq = 2 * sqrt(area / pi)
        d_eq_list_seg.append(d_eq)
    
    
    d_eq_list_g = []
    for ellipse_g in groundtruth_ellipses:
        a_semimajor = ellipse_g[1][1]
        b_semiminor = ellipse_g[1][0]
        area = pi * a_semimajor * b_semiminor
        d_eq = 2 * sqrt(area / pi)
        d_eq_list_g.append(d_eq)
    

    
    return [d_eq_list_seg, d_eq_list_g]

#parse input
parser = argparse.ArgumentParser(description='Code for evaluating Image Segmentation.')
parser.add_argument('seg_csv', help='Path to folder with Segmentation csv.', type=str)
parser.add_argument('groundtruth_csv', help='Path to folder with groundtruth csv.', type=str)
parser.add_argument('output_csv', help='Path & Name to output csv.', type=str)
args = parser.parse_args()
seg_csv_folder = args.seg_csv
groundtruth_csv_folder = args.groundtruth_csv
output_csv = args.output_csv

#load csv files
tp, fp, fn, precision, recall, f1_score, average_jaccard, sauter_seg, sauter_g = ([] for i in range(9))
total_d_eq_seg = []
total_d_eq_g = []
names = []
for groundtruth_csv, seg_csv in zip(sorted(glob.glob('{}/*.csv'.format(groundtruth_csv_folder))), sorted(glob.glob('{}/*.csv'.format(seg_csv_folder)))):
    print("\n")
    print(groundtruth_csv)
    print(seg_csv)
    groundtruth_ellipses = groundtruth(groundtruth_csv)
    seg_output_ellipses = seg_output(seg_csv)
    #calculate sauter
    sauter_list = sauter(groundtruth_ellipses, seg_output_ellipses)
    sauter_seg.append(sauter_list[0])
    sauter_g.append(sauter_list[1])

    #calculate jaccard etc.
    result = jaccard_f1(groundtruth_ellipses, seg_output_ellipses)
    tp.append(result[0])
    fp.append(result[1])
    fn.append(result[2])
    precision.append(result[3])
    recall.append(result[4])
    f1_score.append(result[5])
    average_jaccard.append(result[6])

    #PDF
    d_eq_list = probability_density(groundtruth_ellipses, seg_output_ellipses)
    total_d_eq_seg = total_d_eq_seg + d_eq_list[0]
    total_d_eq_g = total_d_eq_g + d_eq_list[1]
    print(f"length d_eq_seg:{len(total_d_eq_seg)}")
    print(f"length d_eq_g:{len(total_d_eq_g)}")

    #name of current image
    base_name = os.path.basename(seg_csv)
    name = os.path.splitext(base_name)[0]
    names.append(name)
    



#results dictionary
results = {}
results["name"] = names
results["tp"] = tp
results["fp"] = fp
results["fn"] = fn
results["precision"] = precision
results["recall"] = recall
results["f1_score"] = f1_score
results["average_jaccard"] = average_jaccard
results["sauter_seg"] = sauter_seg
results["sauter_g"] = sauter_g
results["mean_precision"] = mean(precision)
results["mean_recall"] = mean(recall)
results["mean_f1_score"] = mean(f1_score)
results["mean_average_jaccard"] = mean(average_jaccard)
results["mean_sauter_seg"] = mean(sauter_seg)
results["mean_sauter_g"] = mean(sauter_g)

# Create Pandas DataFrame
df = pd.DataFrame(results)
print(df)

# Export dataframe to a csv file
df.to_csv(path_or_buf = output_csv, index = None, header=True)

mpl.rcParams.update({'font.size': 15})

density_seg = gaussian_kde(total_d_eq_seg)
density_g = gaussian_kde(total_d_eq_g)


fig = plt.figure()
bins = range(0,200) #not sure what value to choose
                
plt.plot(bins, density_seg(bins), "--", color='red', label='algorithm')
plt.plot(bins, density_g(bins), color='blue', label='ground truth')
plt.legend()
plt.xlabel(r"Equivalent diameter $d_{eq}$")
plt.ylabel("Probability density")
plt.tight_layout()
#plt.show()

print(total_d_eq_g)
print("\n")
print(total_d_eq_seg)
graph_path = os.path.dirname(output_csv)
graph_name = os.path.splitext(os.path.basename(output_csv))[0]
graph = "{0}/{1}".format(graph_path, graph_name)
fig.savefig(f"{graph}.pdf")

fig = plt.figure()

bins = range(0,200,10) #not sure what value to choose

plt.hist(total_d_eq_g, bins=bins, histtype=u'stepfilled', density=False, color='blue', label='ground truth')
plt.hist(total_d_eq_seg, bins=bins, histtype=u'step', density=False, color='red', label='algorithm')
plt.xlabel(r"Equivalent diameter $d_{eq}$", fontsize=15)
plt.ylabel("Frequency", fontsize=15) 
plt.legend()

graph_path = os.path.dirname(output_csv)
graph_name = os.path.splitext(os.path.basename(output_csv))[0]
graph = "{0}/{1}".format(graph_path, graph_name+"_2")
fig.savefig(f"{graph}.pdf")