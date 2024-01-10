import matplotlib.pyplot as plt
import csv
import argparse
import glob
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib as mpl
from matplotlib.pyplot import figure
from sklearn import metrics

def csv_data(csv_file, column):
    column_data = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                column_data.append(eval(row["{}".format(column)]))
    return column_data


algorithm = csv_data("graphics/scatter_smd/concavepoint.csv", "sauter_seg")
algorithm = np.array(algorithm)
groundtruth = csv_data("graphics/scatter_smd/concavepoint.csv", "sauter_g")
groundtruth = np.array(groundtruth)

mpl.rcParams.update({'font.size': 15})

N = len(algorithm)
colors = np.random.rand(N)

fig, ax = plt.subplots(figsize=(5,5))

# Fit with polyfit
b, m = polyfit(groundtruth, algorithm, 1)

#RMSE
rmse = np.sqrt(metrics.mean_squared_error(groundtruth, algorithm))
print(rmse)

#r_squared https://www.kite.com/python/answers/how-to-calculate-r-squared-with-numpy-in-python
correlation_matrix = np.corrcoef(groundtruth, algorithm)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
r_squared = round(r_squared, 3)
r_squared = r"$R^2 =$ {}".format(r_squared)

rmse = round(rmse, 2)
rmse = r"$RMSE =$ {}".format(rmse)
plt.text(138, 190, rmse)


all_data = np.append(groundtruth, algorithm)
plt.xlim(min(all_data), max(all_data))
plt.ylim(min(all_data), max(all_data))

plt.xticks(np.arange(min(all_data).round(-1), (max(all_data)+1).round(-1), 20.0))
plt.yticks(np.arange(min(all_data).round(-1), (max(all_data)+1).round(-1), 20.0))


plt.scatter(groundtruth, algorithm,  c="red")
#ax.plot([0, 1.15], [0, 1], "-.", transform=ax.transAxes, color="black", label="-15%", linewidth=0.8)
#ax.plot([0, 1], [0, 1.15], "--", transform=ax.transAxes, color="black", label="-15%", linewidth=0.8)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", linewidth=2)
plt.plot([min(groundtruth), max(groundtruth)], [b + m * min(groundtruth), b + m * max(groundtruth)] , '--', label="linear regression", linewidth=2)
ax.legend()
ax.set_xlabel("SMD groundtruth")
ax.set_ylabel("SMD algorithm")

plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
#plt.subplots_adjust(bottom=0.1)
plt.savefig('graphics/scatter_smd/concavepoint.pdf')
plt.show()



#fig.savefig(f"{name_graph}.pdf")