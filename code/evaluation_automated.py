import glob
import cv2
import argparse
import os
import pandas as pd
import operator
import csv
from collections import OrderedDict
import glob

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
#list of values per parameter that need to be tested
value_list = {
    "mindis": [0]
}
value_list = OrderedDict(value_list.items()) #convert to ordered dictionary

#list of initial parameter values that get changed to best values throughout
param_list = {
    "median": 11,
    "blocksize": 17,
    "constant": 3,
    "ksize": 5,
    "oksize": 7,
    "mindis": 0
}
param_list = OrderedDict(param_list.items()) #convert to ordered dictionary

def csv_data(csv_file, column):
    column_data = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                column_data.append(eval(row["{}".format(column)]))
    return column_data


for parameter, values in value_list.items():
    param_folder = output_folder + parameter
    try:
        os.mkdir(param_folder) #create dictionary for parameter
    except:
        print("already exists")
    
    param_val_settings_folder = output_folder + parameter + "_settings"

    try:
        os.mkdir(param_val_settings_folder) #create folder for settings log that were used
    except:
        print("already exists")


    for val in values: #loop values for each parameter and run algorithm
        print(val)
        
        param_list[parameter] = val

        params = [str(x) for x in param_list.values()]

        algo_params = " ".join(params[5:])  #convert to string form
        thresh_params = " ".join(params[:5])
        

        folder = val
        
        directory = param_folder + "/{}/".format(folder)
        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory {} failed".format(directory))
        else:
            print ("Successfully created the directory {} ".format(directory))

        
        for idx, filename in enumerate(glob.glob('{}/*.png'.format(input_image_folder))):
            print("python {0}.py {1} {2} --thresh_params {3} --ws_params {4}".format(algorithm, filename, directory, thresh_params, algo_params))
            os.system("python {0}.py {1} {2} --thresh_params {3} --ws_params {4}".format(algorithm, filename, directory, thresh_params, algo_params))

        settings["algo_params"] = algo_params
        settings["thresh_params"] = thresh_params


        df = pd.DataFrame(settings, index=[0])

        # Export dataframe to a csv file

        df.to_csv(path_or_buf = param_val_settings_folder + "/" + parameter + "_{}".format(val) + "_settings.csv", index = None, header=True)
            

    #evaluate parameter, select best one, and change settings
    
    param_val_results_folder = output_folder + parameter + "_results"
    try:
        os.mkdir(param_val_results_folder)
    except:
        print("already exists")
    print("python run_evaluation.py {0} run2/truth/ {1}/ ".format(param_folder, param_val_results_folder))
    os.system("python run_evaluation.py {0} run2/truth/ {1}/ ".format(param_folder, param_val_results_folder)) #WATCH OUT: fixed truth folder

    #determine best value
    files = []
    for entry in glob.glob('{}/*.csv'.format(param_val_results_folder)):
        files.append(entry)

    mean_f1_score = {}
    for fl in files:
        mean_f1_score["{}".format(int(os.path.basename(os.path.splitext(fl)[0])))] = csv_data(fl, "mean_f1_score")[0]
    best_value = max(mean_f1_score, key=lambda key: mean_f1_score[key])
    print(best_value)

    param_list[parameter] = best_value #change value to best candidate

print(param_list)

"""
for param in [0]:
    print(param)

    algo_params = "1 26 1 9 0 100".format(param)
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

    df.to_csv(path_or_buf = output_folder + parameter + "_{}".format(param) + "_settings.csv", index = None, header=True)
"""





   


