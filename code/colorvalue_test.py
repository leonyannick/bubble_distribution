import cv2
import numpy as np
import os

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        #colorsB = image[y,x,0]
        #colorsG = image[y,x,1]
        #colorsR = image[y,x,2]
        colors = image[y,x]
        #print("Red: ",colorsR)
        #print("Green: ",colorsG)
        #print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

# Read an image, a window and bind the function to window
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
print(script_dir)
rel_path = 'pictures/c_P1260783.png'
abs_file_path = os.path.join(script_dir, rel_path)
image = cv2.imread(abs_file_path, cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('mouseRGB', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('mouseRGB',mouseRGB)

sz = 1
image = cv2.resize(image, (0,0), fx=sz, fy=sz)
#Do until esc pressed
while(1):
    cv2.imshow('mouseRGB',image)
    if cv2.waitKey(20) & 0xFF == 32:
        break
#if esc pressed, finish.
cv2.destroyAllWindows()