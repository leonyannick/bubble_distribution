from math import log10, sqrt 
import cv2 
import numpy as np 
import glob
from statistics import mean
  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main():


    original = cv2.imread("pictures/c_P1260782.png") 
    compressed = cv2.imread("pictures/median_test.png", 1) 
    value = PSNR(original, compressed) 
    #print(f"PSNR value is {value} dB") 
    
if __name__ == "__main__": 
    main() 


fltr_path = "run1/cp/md_psnr/5/"
truth_path = "run1/input/"
values_list = []
for flter, truth in zip(glob.glob('{}/*.png'.format(fltr_path)), glob.glob('{}/*.png'.format(truth_path)) ):
    original = cv2.imread(truth) 
    compressed = cv2.imread(flter) 
    value = PSNR(original, compressed)
    values_list.append(value)
    
mean_psnr = mean(values_list)
print(mean_psnr)

    