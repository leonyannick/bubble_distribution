import matplotlib.pyplot as plt 
import cv2
import numpy as np
import argparse
import os
import csv
import glob
import re
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


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

"""
#single plot

files = []
for entry in glob.glob('{}/*.csv'.format(input_folder)):
    files.append(entry)


sort_nicely(files)


mean_f1_score = []
mean_AJSC = []
delta_SMD = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
    mean_AJSC.append(csv_data(fl, "mean_average_jaccard")[0])
    delta_SMD.append(abs(csv_data(fl, "mean_sauter_seg")[0] - csv_data(fl, "mean_sauter_g")[0]))



x = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
print(len(x))
print(x)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))


xlabel = os.path.basename(os.path.normpath(input_folder))

#f1
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = x[f1_xpos]
#ax[0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax+0.5*f1_xmax, f1_ymax),arrowprops=dict(facecolor='black', shrink=0.05))
#f1 = ax[0].plot(x, mean_f1_score, "ro-", color='red', label="f1-score")

ax.annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.1),arrowprops=dict(facecolor='black', shrink=0.1))
f1 = ax.plot(x, mean_f1_score, ".b", label="f1-score")
ax.set_ylim([0,1])
ax.set_xlabel(f"{xlabel}")
ax.set_ylabel("f1-score")

#ax.set_xlim(xmin=0)
#ticks x axis
ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in', top='on')
#ax.xaxis.set_tick_params(which='minor', size=3.5, width=1, direction='in', top='on')
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
#ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

#ticks y axis
ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in', right='on')
#ax.yaxis.set_tick_params(which='minor', size=3.5, width=1, direction='in', right='on')
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
#ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))

plt.tight_layout()
plt.show()
name_graph = os.path.normpath(input_folder)
fig.savefig(f"{name_graph}.pdf")
  """  

#grid of plots for parameter selection hough
mpl.rcParams.update({'font.size': 17})
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11.69,8.27), sharey=True)



#median
median = input_folder + "/median/"
files = []
for entry in glob.glob('{}/*.csv'.format(median)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
kernel_size = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = kernel_size[f1_xpos]
ax[0, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 0].set_ylim([0,1])
ax[0, 0].set_xlabel(r"$K_{median}$")
ax[0, 0].set_ylabel("f1-score")
ax[0, 0].plot(kernel_size, mean_f1_score, ".r", label="f1-score")
ax[0, 0].tick_params(direction="in", width=2)
ax[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#blocksize
blocksize = input_folder + "/blocksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(blocksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
blocksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = blocksize[f1_xpos]
ax[0, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 1].set_ylim([0,1])
ax[0, 1].set_xlabel(r"$K_{T}$")

ax[0, 1].plot(blocksize, mean_f1_score, ".r", label="f1-score")
ax[0, 1].tick_params(direction="in", width=2)
ax[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#blocksize
constant = input_folder + "/constant/"
files = []
for entry in glob.glob('{}/*.csv'.format(constant)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
constant = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = constant[f1_xpos]
ax[0, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 2].set_ylim([0,1])
ax[0, 2].set_xlabel(r"$c$")

ax[0, 2].plot(constant, mean_f1_score, ".r", label="f1-score")
ax[0, 2].tick_params(direction="in", width=2)
ax[0, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
#ksize
ksize = input_folder + "/ksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(ksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
ksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = ksize[f1_xpos]
ax[1, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 0].set_ylim([0,1])
ax[1, 0].set_xlabel(r"$K_{dil/ero}$")
ax[1, 0].set_ylabel("f1-score")
ax[1, 0].plot(ksize, mean_f1_score, ".r", label="f1-score")
ax[1, 0].tick_params(direction="in", width=2)
ax[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#oksize
oksize = input_folder + "/oksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(oksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
oksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = oksize[f1_xpos]
ax[1, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 1].set_ylim([0,1])
ax[1, 1].set_xlabel(r"$K_{op}$")

ax[1, 1].plot(oksize, mean_f1_score, ".r", label="f1-score")
ax[1, 1].tick_params(direction="in", width=2)
ax[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#mindis
mindis = input_folder + "/mindis/"
files = []
for entry in glob.glob('{}/*.csv'.format(mindis)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
mindis = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = mindis[f1_xpos]
ax[1, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 2].set_ylim([0,1])
ax[1, 2].set_xlabel(r"$minDis$")

ax[1, 2].plot(mindis, mean_f1_score, ".r", label="f1-score")
ax[1, 2].tick_params(direction="in", width=2)
ax[1, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
#param2
mindis = input_folder + "/param2/"
files = []
for entry in glob.glob('{}/*.csv'.format(mindis)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
mindis = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = mindis[f1_xpos]
ax[2, 0].set_ylabel("f1-score")
ax[2, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[2, 0].set_ylim([0,1])
ax[2, 0].set_xlabel(r"$T_{acc}$")
ax[2, 0].plot(mindis, mean_f1_score, ".r", label="f1-score")
ax[2, 0].tick_params(direction="in", width=2)
ax[2, 0].xaxis.set_major_locator(MaxNLocator(integer=True))


ax[2,1].axis('off')
ax[2,2].axis('off')


fig.subplots_adjust(left=0.07, bottom=0.1, right=0.92, top=0.98, wspace=0.15, hspace=0.26)
#show and save graph
#plt.tight_layout()
plt.show()
name_graph = os.path.normpath(input_folder)
fig.savefig(f"{name_graph}.pdf")

"""
#grid of plots for parameter selection watershed
mpl.rcParams.update({'font.size': 17})
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11.69,8.27), sharey=True)



#median
median = input_folder + "/median/"
files = []
for entry in glob.glob('{}/*.csv'.format(median)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
kernel_size = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = kernel_size[f1_xpos]
ax[0, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 0].set_ylim([0,1])
ax[0, 0].set_xlabel(r"$K_{median}$")
ax[0, 0].set_ylabel("f1-score")
ax[0, 0].plot(kernel_size, mean_f1_score, ".r", label="f1-score")
ax[0, 0].tick_params(direction="in", width=2)
ax[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#blocksize
blocksize = input_folder + "/blocksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(blocksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
blocksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = blocksize[f1_xpos]
ax[0, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 1].set_ylim([0,1])
ax[0, 1].set_xlabel(r"$K_{T}$")

ax[0, 1].plot(blocksize, mean_f1_score, ".r", label="f1-score")
ax[0, 1].tick_params(direction="in", width=2)
ax[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#blocksize
constant = input_folder + "/constant/"
files = []
for entry in glob.glob('{}/*.csv'.format(constant)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
constant = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = constant[f1_xpos]
ax[0, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 2].set_ylim([0,1])
ax[0, 2].set_xlabel(r"$c$")

ax[0, 2].plot(constant, mean_f1_score, ".r", label="f1-score")
ax[0, 2].tick_params(direction="in", width=2)
ax[0, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
#ksize
ksize = input_folder + "/ksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(ksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
ksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = ksize[f1_xpos]
ax[1, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 0].set_ylim([0,1])
ax[1, 0].set_xlabel(r"$K_{dil/ero}$")
ax[1, 0].set_ylabel("f1-score")
ax[1, 0].plot(ksize, mean_f1_score, ".r", label="f1-score")
ax[1, 0].tick_params(direction="in", width=2)
ax[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#oksize
oksize = input_folder + "/oksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(oksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
oksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = oksize[f1_xpos]
ax[1, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 1].set_ylim([0,1])
ax[1, 1].set_xlabel(r"$K_{op}$")

ax[1, 1].plot(oksize, mean_f1_score, ".r", label="f1-score")
ax[1, 1].tick_params(direction="in", width=2)
ax[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#mindis
mindis = input_folder + "/mindis/"
files = []
for entry in glob.glob('{}/*.csv'.format(mindis)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
mindis = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = mindis[f1_xpos]
ax[1, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 2].set_ylim([0,1])
ax[1, 2].set_xlabel(r"$minDis$")

ax[1, 2].plot(mindis, mean_f1_score, ".r", label="f1-score")
ax[1, 2].tick_params(direction="in", width=2)
ax[1, 2].xaxis.set_major_locator(MaxNLocator(integer=True))


fig.subplots_adjust(left=0.07, bottom=0.1, right=0.92, top=0.98, wspace=0.15, hspace=0.26)
#show and save graph
#plt.tight_layout()
plt.show()
name_graph = os.path.normpath(input_folder)
fig.savefig(f"{name_graph}.pdf")
"""
"""
#grid of plots for parameter selection concavepoint
mpl.rcParams.update({'font.size': 17})
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11.69,8.27), sharey=True)



#median
median = input_folder + "/median/"
files = []
for entry in glob.glob('{}/*.csv'.format(median)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
kernel_size = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = kernel_size[f1_xpos]
ax[0, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 0].set_ylim([0,1])
ax[0, 0].set_xlabel(r"$K_{median}$")
ax[0, 0].set_ylabel("f1-score")
ax[0, 0].plot(kernel_size, mean_f1_score, ".r", label="f1-score")
ax[0, 0].tick_params(direction="in", width=2)
ax[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#blocksize
blocksize = input_folder + "/blocksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(blocksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
blocksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = blocksize[f1_xpos]
ax[0, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 1].set_ylim([0,1])
ax[0, 1].set_xlabel(r"$K_{T}$")

ax[0, 1].plot(blocksize, mean_f1_score, ".r", label="f1-score")
ax[0, 1].tick_params(direction="in", width=2)
ax[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#blocksize
constant = input_folder + "/constant/"
files = []
for entry in glob.glob('{}/*.csv'.format(constant)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
constant = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = constant[f1_xpos]
ax[0, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[0, 2].set_ylim([0,1])
ax[0, 2].set_xlabel(r"$c$")

ax[0, 2].plot(constant, mean_f1_score, ".r", label="f1-score")
ax[0, 2].tick_params(direction="in", width=2)
ax[0, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
#ksize
ksize = input_folder + "/ksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(ksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
ksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = ksize[f1_xpos]
ax[1, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 0].set_ylim([0,1])
ax[1, 0].set_xlabel(r"$K_{dil/ero}$")
ax[1, 0].set_ylabel("f1-score")
ax[1, 0].plot(ksize, mean_f1_score, ".r", label="f1-score")
ax[1, 0].tick_params(direction="in", width=2)
ax[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#oksize
oksize = input_folder + "/oksize/"
files = []
for entry in glob.glob('{}/*.csv'.format(oksize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
oksize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = oksize[f1_xpos]
ax[1, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 1].set_ylim([0,1])
ax[1, 1].set_xlabel(r"$K_{op}$")

ax[1, 1].plot(oksize, mean_f1_score, ".r", label="f1-score")
ax[1, 1].tick_params(direction="in", width=2)
ax[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#angle
angle = input_folder + "/angle/"
files = []
for entry in glob.glob('{}/*.csv'.format(angle)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
angle = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = angle[f1_xpos]
ax[1, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[1, 2].set_ylim([0,1])
ax[1, 2].set_xlabel(r"$T_{\theta}$")

ax[1, 2].plot(angle, mean_f1_score, ".r", label="f1-score")
ax[1, 2].tick_params(direction="in", width=2)
ax[1, 2].xaxis.set_major_locator(MaxNLocator(4))

#stepsize
stepsize = input_folder + "/stepsize/"
files = []
for entry in glob.glob('{}/*.csv'.format(stepsize)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
stepsize = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = stepsize[f1_xpos]
ax[2, 0].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[2, 0].set_ylim([0,1])
ax[2, 0].set_xlabel(r"$stepsize$")

ax[2, 0].plot(stepsize, mean_f1_score, ".r", label="f1-score")
ax[2, 0].tick_params(direction="in", width=2)
ax[2, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
#mindis_cp
mindis_cp = input_folder + "/mindis_cp/"
files = []
for entry in glob.glob('{}/*.csv'.format(mindis_cp)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
mindis_cp = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = mindis_cp[f1_xpos]
ax[2, 1].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[2, 1].set_ylim([0,1])
ax[2, 1].set_xlabel(r"$mindis_{cp}$")

ax[2, 1].plot(mindis_cp, mean_f1_score, ".r", label="f1-score")
ax[2, 1].tick_params(direction="in", width=2)
ax[2, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
#mindis_ellipses
mindis_ellipses = input_folder + "/mindis_ellipses/"
files = []
for entry in glob.glob('{}/*.csv'.format(mindis_ellipses)):
    files.append(entry)
sort_nicely(files)
mean_f1_score = []
for fl in files:
    mean_f1_score.append(csv_data(fl, "mean_f1_score")[0])
mindis_ellipses = [int(os.path.basename(os.path.splitext(x)[0])) for x in files]
f1_ymax = max(mean_f1_score)
f1_xpos = mean_f1_score.index(f1_ymax)
f1_xmax = mindis_ellipses[f1_xpos]
ax[2, 2].annotate(f'max: {f1_xmax}', xy=(f1_xmax, f1_ymax), xytext=(f1_xmax, f1_ymax-0.3),arrowprops=dict(facecolor='black', shrink=0.1, width=2))
ax[2, 2].set_ylim([0,1])
ax[2, 2].set_xlabel(r"$mindis_{ellipses}$")

ax[2, 2].plot(mindis_ellipses, mean_f1_score, ".r", label="f1-score")
ax[2, 2].tick_params(direction="in", width=2)
ax[2, 2].xaxis.set_major_locator(MaxNLocator(integer=True))


fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.99, wspace=0.12, hspace=0.29)
#show and save graph
#plt.tight_layout()
plt.show()
name_graph = os.path.normpath(input_folder)
fig.savefig(f"{name_graph}.pdf")
"""
"""
#ajsc
ajsc_ymax = max(mean_AJSC)
ajsc_xpos = mean_AJSC.index(ajsc_ymax)
ajsc_xmax = x[ajsc_xpos]
ax[0].annotate(f'max: {ajsc_xmax}', xy=(ajsc_xmax, ajsc_ymax), xytext=(ajsc_xmax+0.5*ajsc_xmax, ajsc_ymax),arrowprops=dict(facecolor='black', shrink=0.05))
ajsc = ax[0].plot(x, mean_AJSC, "ro-", color='blue', label="AJSC")

ax[0].set_xlabel(f"{xlabel}")
ax[0].set_ylabel("f1-score/AJSC")
ax[0].legend(loc='lower right')


#smd
smd_ymax = min(delta_SMD)
smd_xpos = delta_SMD.index(smd_ymax)
smd_xmax = x[smd_xpos]
ax[1].annotate(f'max: {smd_xmax}', xy=(smd_xmax, smd_ymax), xytext=(smd_xmax+0.5*smd_xmax, smd_ymax),arrowprops=dict(facecolor='black', shrink=0.05))
smd = ax[1].plot(x, delta_SMD, "ro-", color='red')

ax[1].set_xlabel(f"{xlabel}")
ax[1].set_ylabel(f"f1-score")
"""
