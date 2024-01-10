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
import glob


csv_files = glob.glob("evaluation/ellipse_generator_calibration_2/*.csv")
print(csv_files)

axes = []
angle = []
ellipses_list = []
for csv_file in csv_files:
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ellipses_list.append(eval(row["ellipses_list"]))
 
for ellipse in ellipses_list:
    angle.append(ellipse[2])
    axes.append(ellipse[1])



data = {}
data["axes"] = axes
data["angle"] = angle
# Create the Pandas DataFrame
df = pd.DataFrame(data)

# Export the dataframe to a csv file
path = 'evaluation/ellipse_generator_calibration_2/data.csv'
df.to_csv(path_or_buf = path , index = None, header=True)

