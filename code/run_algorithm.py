import glob
import cv2
import argparse
import os
import pandas as pd

#command-line input
parser = argparse.ArgumentParser(description='Code for running Image Segmentation algorithms.')
parser.add_argument('input_image_folder', help='Path to input image folder.', type=str)
parser.add_argument('output_folder', help='Path to output image folder.', type=str)
parser.add_argument('algorithm', help='Name of algorithm.', type=str)
args = parser.parse_args()

input_image_folder = args.input_image_folder
output_folder = args.output_folder
algorithm = args.algorithm

settings = {}
for param in [0]:
    print(param)

    algo_params = "1 26 1 {} 0 100".format(param)
    thresh_params ="5 17 6 0 7".format(param)
    folder = param
    



    directory = output_folder + "{}/".format(folder)
    try:
        os.mkdir(directory)
    except OSError:
        print ("Creation of the directory {} failed".format(directory))
    else:
        print ("Successfully created the directory {} ".format(directory))

    
    for idx, filename in enumerate(glob.glob('{}/*.png'.format(input_image_folder))):
        print("python {0}.py {1} {2} --thresh_params {3} --hough_params {4}".format(algorithm, filename, directory, thresh_params, algo_params))
        os.system("python {0}.py {1} {2} --thresh_params {3} --hough_params {4}".format(algorithm, filename, directory, thresh_params, algo_params))

    settings["algo_params"] = algo_params
    settings["thresh_params"] = thresh_params


    df = pd.DataFrame(settings, index=[0])

    # Export dataframe to a csv file

    df.to_csv(path_or_buf = output_folder+"{}".format(param)+"_settings.csv", index = None, header=True)







   


