import glob
import cv2
import argparse
import os
import glob


#command-line input
parser = argparse.ArgumentParser(description='Code for evaluating Image Segmentation.')
parser.add_argument('directory', help='Path to directory of output directories.', type=str)
parser.add_argument('groundtruth_csv', help='Path to folder with groundtruth csv.', type=str)
parser.add_argument('output_folder', help='Path to output image folder.', type=str)
args = parser.parse_args()

directory = args.directory
output_folder = args.output_folder
groundtruth_csv_folder = args.groundtruth_csv

for subdir, dirs, files in os.walk(directory):
    name = os.path.basename(subdir)
    output_csv = output_folder + name + ".csv"
    os.system("python evaluation.py {0} {1} {2}".format(subdir, groundtruth_csv_folder, output_csv))


"""
for idx, filename in enumerate(glob.glob('{}/*.png'.format(input_image_folder))):
    os.system("python {0}.py {1} {2} --thresh_params {3} 17 3 5 7".format(algorithm, filename, directory, param))
    print("python {0}.py {1} {2} --thresh_params {3} 17 3 5 7".format(algorithm, filename, directory, param))
"""
