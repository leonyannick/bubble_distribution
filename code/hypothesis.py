import csv
import numpy as np
from math import sqrt


def csv_data(csv_file, column):
    column_data = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                column_data.append(eval(row["{}".format(column)]))
    return column_data

csv_file = "graphics/hypothesentest/hough.csv"

f1_score = csv_data(csv_file, "f1_score")
print(f1_score)

sample_size = 50
mu_0 = 0.9
mean = np.mean(f1_score)

covariance = np.cov(f1_score)
print("covariance:{}".format(covariance))

standard_deviation = np.std(f1_score)
print("standard_deviation:{}".format(standard_deviation))

T = ((mean - mu_0) / standard_deviation) * sqrt(sample_size)
print("T:{}".format(T))

from_table = 1.64