import matplotlib.pyplot as plt 
import cv2
import numpy as np
import argparse
import os
import csv
import glob
import re

import math 
import copy 

class Point(): 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# A utility function to find the  
# distance between two points  
def dist(p1, p2): 
    return math.sqrt((p1.x - p2.x) * 
                     (p1.x - p2.x) +
                     (p1.y - p2.y) * 
                     (p1.y - p2.y))  
  
# A Brute Force method to return the  
# smallest distance between two points  
# in P[] of size n 
def bruteForce(P, n): 
    min_val = float('inf')  
    for i in range(n): 
        for j in range(i + 1, n): 
            if dist(P[i], P[j]) < min_val: 
                min_val = dist(P[i], P[j]) 
  
    return min_val 
  
# A utility function to find the  
# distance beween the closest points of  
# strip of given size. All points in  
# strip[] are sorted accordint to  
# y coordinate. They all have an upper  
# bound on minimum distance as d.  
# Note that this method seems to be  
# a O(n^2) method, but it's a O(n)  
# method as the inner loop runs at most 6 times 
def stripClosest(strip, size, d): 
      
    # Initialize the minimum distance as d  
    min_val = d  
  
     
    # Pick all points one by one and  
    # try the next points till the difference  
    # between y coordinates is smaller than d.  
    # This is a proven fact that this loop 
    # runs at most 6 times  
    for i in range(size): 
        j = i + 1
        while j < size and (strip[j].y - 
                            strip[i].y) < min_val: 
            min_val = dist(strip[i], strip[j]) 
            j += 1
  
    return min_val  
  
# A recursive function to find the  
# smallest distance. The array P contains  
# all points sorted according to x coordinate 
def closestUtil(P, Q, n): 
      
    # If there are 2 or 3 points,  
    # then use brute force  
    if n <= 3:  
        return bruteForce(P, n)  
  
    # Find the middle point  
    mid = n // 2
    midPoint = P[mid] 
  
    # Consider the vertical line passing  
    # through the middle point calculate  
    # the smallest distance dl on left  
    # of middle point and dr on right side  
    dl = closestUtil(P[:mid], Q, mid) 
    dr = closestUtil(P[mid:], Q, n - mid)  
  
    # Find the smaller of two distances  
    d = min(dl, dr) 
  
    # Build an array strip[] that contains  
    # points close (closer than d)  
    # to the line passing through the middle point  
    strip = []  
    for i in range(n):  
        if abs(Q[i].x - midPoint.x) < d:  
            strip.append(Q[i]) 
  
    # Find the closest points in strip.  
    # Return the minimum of d and closest  
    # distance is strip[]  
    return min(d, stripClosest(strip, len(strip), d)) 
  
# The main function that finds 
# the smallest distance.  
# This method mainly uses closestUtil() 
def closest(P, n): 
    P.sort(key = lambda point: point.x) 
    Q = copy.deepcopy(P) 
    Q.sort(key = lambda point: point.y)     
  
    # Use recursive function closestUtil()  
    # to find the smallest distance  
    return closestUtil(P, Q, n) 


def csv_data(csv_file, column):
    column_data = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                column_data.append(eval(row["{}".format(column)]))
    return column_data

parser = argparse.ArgumentParser(description='Code for graphing stuff.')
parser.add_argument('input_folder', help='Path to input folder.', type=str)
args = parser.parse_args()
input_folder = args.input_folder



files = []

for entry in os.scandir(input_folder):
    entry_path = (entry.path)
    files.append(entry_path)





center = []
for fl in files:
    center.append(csv_data(fl, "center"))




P = []
for point in center[0]:
    P.append(Point(point[0],point[1]))


n = len(P)  
print("The smallest distance is",  closest(P, n)) 
print("test", n)