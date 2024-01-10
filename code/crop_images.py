import cv2
import argparse
import glob
import os
import random



def main():
    #command-line input
    parser = argparse.ArgumentParser(description='Code for evaluating Image Segmentation.')
    parser.add_argument('input_image_folder', help='Path to input image folder.', type=str)
    parser.add_argument('output_image_folder', help='Path to output image folder.', type=str)
    args = parser.parse_args()

    input_image_folder = args.input_image_folder
    output_image_folder = args.output_image_folder

    #cropping parameters
    
    width = 2000
    height = 1650
    #loops through input images
    for filename in glob.glob('{}/*.png'.format(input_image_folder)): #assuming png
        x_start = int(random.uniform(0, 3200))
        y_start = int(random.uniform(0, 2254))
         # Read image
        file_basename = os.path.basename(filename)
        im = cv2.imread(filename)

        #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        # Select ROI
        #fromCenter = False
        #r = cv2.selectROI("Image", im, fromCenter)
        
        # Crop image
        imCrop = im[y_start:y_start + height , x_start:x_start + width]
        cv2.imwrite("{0}/{1}".format(output_image_folder, "c_" + file_basename), imCrop)


        # Display cropped image
        #cv2.imshow("Image", imCrop)
        #cv2.waitKey(0)
        


if __name__ == "__main__":
    main()     