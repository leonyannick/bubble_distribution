import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import pandas as pd # Import Pandas library
import sys # Enables the passing of arguments
from itertools import chain
import argparse
import glob
import os
from math import sqrt, pi
from sklearn.metrics import jaccard_score
from shapely.geometry.point import Point
from shapely.geometry import Polygon
from shapely import affinity
from statistics import mean
#URL: https://automaticaddison.com/how-to-annotate-images-using-opencv/

#terminal: cd c:/Users/Leon/Desktop/Code_BA/bachelorarbeit/
#python .\auswertung.py Beispielbilder/2_labor.jpg Beispielbilder/data.csv

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, (lengths[0] / 2), (lengths[1] / 2))
    ellr = affinity.rotate(ell, angle)
    if not ellr.is_valid:
        return 0
    return ellr
 
# Define the file name of the image
parser = argparse.ArgumentParser(description='Code for marking images by hand.')
parser.add_argument('input_image', help='Path to input image.', type=str)
parser.add_argument('output_image_folder', help='Path to output image folder.', type=str)
parser.add_argument('output_data_folder', help='Path to output data file folder.', type=str)
args = parser.parse_args()

input_image = args.input_image
output_image_folder = args.output_image_folder
output_data_folder = args.output_data_folder
"""
#loops through input images"
for filename in glob.glob('{}/*.png'.format(input_image_folder)): #assuming png
    #os.system("python mark.py {0} {1} {2}".format(filename, output_image_folder, output_data_folder))
    #print("python mark.py {0} {1} {2}".format(filename, output_image_folder, output_data_folder))
    INPUT_IMAGE = 
    print(os.path.basename(filename))
"""

base_name = os.path.basename(input_image)
base_name_stripped = base_name[:base_name.index(".")]
output_image = output_image_folder + "/a_" + base_name_stripped + ".png"
output_data = output_data_folder + "/" + base_name_stripped + ".csv"
# Load the image and store into a variable
# -1 means load unchanged
image = cv2.imread(input_image, -1)
gray = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
# Create lists to store all x, y, and annotation values
coordinates = []
coordinates2 = []
x_vals = []
y_vals = []
annotation_vals = []

#size of image
sz = 1

#number of segmented bubbles
number = 0

#create dictionary
data = {}
 

drawing = False # true if mouse is pressed
mode = False # default is drawing circles, press m to switch to rextangles
ix,iy = -1,-1 
def draw_circle(event, x, y, flags, param):
    """
    Draws dots on double clicking of the left mouse button
    """
    # Store the height and width of the image
    global ix, iy, drawing, mode
    height = image.shape[0]
    width = image.shape[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        xorg = int(x/sz)
        yorg = int(y/sz)
        drawing = True
        ix,iy = x,y
        x_vals.clear()
        y_vals.clear()
        annotation_vals.clear()
        coordinates.clear()
        draw_circle.counter = 0
        txt = str(draw_circle.counter)
        coordinates.append([xorg,yorg])
        coordinates2.append([x,y])
        x_vals.append(xorg)
        y_vals.append(yorg)
        annotation_vals.append(txt)
    elif event == cv2.EVENT_MOUSEMOVE:
        xorg = int(x/sz)
        yorg = int(y/sz)
        if drawing == True:
            if mode == True:
                cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),-1)
                
            else:
                cv2.circle(image,(x,y),1,(0,0,255),-1)
                draw_circle.counter += 1
                txt = str(draw_circle.counter)
                # Append values to the list
                coordinates.append([xorg,yorg])
                coordinates2.append([x,y])
                x_vals.append(xorg)
                y_vals.append(yorg)
                annotation_vals.append(txt)
        

    elif event == cv2.EVENT_LBUTTONUP:
        
        xorg = int(x/sz)
        yorg = int(y/sz)
        drawing = False
        if mode == True:
            cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),-1)
        #else:
            #cv2.circle(image,(x,y),3,(0,0,255),-1)

 
print('functionality: SPACE -> mark bubble with mouse -> S -> F\n')
print("PRESS SPACE TO START SEGMENTATION\n")
print('when finished press ESC\n\n')
 
# We create a named window where the mouse callback will be established
cv2.namedWindow('Image mouse', cv2.WINDOW_NORMAL)
 
# We set the mouse callback function to 'draw_circle':
cv2.setMouseCallback('Image mouse', draw_circle)
backup = [] #copies image in a backup in order to undo drawing mistakes

results = {}
jaccard_score_list = [] #saves all JSC values for AJSC calculation
counter = -1
area = []
perimeter = []
center = []
count = []
a_semimajor = []
b_semiminor = []
eccentricity = []
image = cv2.resize(image, (0,0), fx=sz, fy=sz)
ellipses_list = []
#create mask
angle_ver = []
d_eq = []
while True:
    # Show image 'Image mouse':
    
    cv2.imshow('Image mouse', image)

    # Continue until 'esc' is pressed:
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('m'):
        mode = not mode
    elif k == 32: #SPACE: new semgmentation
        print('starting new segmentation')
        #backup = image.copy()
    elif k == ord('s'): #S: save coordinates in dictionary
        print('saving coordinates')
        data[number] = tuple(coordinates)
        
        
    elif k == ord('f'): #finish segmentation and start examination
        print('calculating parameters')
        print('')
        print('press SPACE')
        contour = list(data[0])
        contour = np.array(contour).reshape((-1,1,2)).astype(np.int32)

        """
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center.append((cx, cy))
        area.append(cv2.contourArea(contour))
        perimeter.append(cv2.arcLength(contour, True))
        """
        counter += 1
        count.append(counter)
        """
        #draw center
        xcropped = int(center[counter][0] * sz)
        ycropped = int(center[counter][1] * sz)
        print(xcropped)
        print(ycropped)
        print(center[counter])
        cv2.circle(image, (xcropped,ycropped), 1, (0,255,0), -1)
        """
        #draw contour
        
        coordinatescropped = [np.array(coordinates2).reshape((-1,1,2)).astype(np.int32)]
        contourzoom = np.array([[element for element in contour]])
        #print([int(b[0]) for a in contourzoom for b in a])
        #print(contourzoom)
        cv2.drawContours(image, coordinatescropped, 0, (255,0,0), 1, cv2.LINE_AA)
        if len(coordinatescropped[0]) > 5:
            ellipse = cv2.fitEllipseDirect(coordinatescropped[0])
            ellipses_list.append(ellipse)
            cv2.ellipse(image, ellipse, [0,0,255], 2)
            center_x = ellipse[0][0]
            center_y = ellipse[0][1]
            c = (center_x, center_y)
            cv2.circle(image, (int(c[0]),int(c[1])), 1, (0,255,0), -1)
            center.append(c)
            b = ellipse[1][0] / 2
            a = ellipse[1][1] / 2
            a_semimajor.append(a)
            b_semiminor.append(b)
            h = (a-b) ** 2 / (a+b) ** 2
            area_ellipse = pi * a  * b
            area.append(area_ellipse)
            perimeter.append(pi * (a + b) * (1 + 3 * h / (10 + sqrt(4 - 3 * h)))) #ramanujan perimeter approx.
            d_eq.append(2 * sqrt(area_ellipse / pi)) #equivalent diameter
            e = sqrt(1 - (b ** 2 / a ** 2))
            eccentricity.append(e)
            angle_ver.append(ellipse[2])
        
            #area comparison between ellipse and contour
            
            ellipse_polygon = create_ellipse(ellipse[0], ellipse[1], ellipse[2])

            contour = np.squeeze(coordinatescropped)
            contur_polygon = Polygon(contour)

            area_ellipse = ellipse_polygon.area
            area_polygon = contur_polygon.area

            try:
                intersect = ellipse_polygon.intersection(contur_polygon)
                area_intersection = intersect.area
                area_union = area_ellipse + area_polygon - area_intersection
                score = area_intersection / area_union

                print(score)
                jaccard_score_list.append(score)
            except:
                print("self intersection, mark again")
            

        coordinates2 = []
        #backup
        backup = image.copy()

    elif k == 8:#BACKSPACE: delete
        print('deleted last segmentation')
        coordinates2 = []
        image = backup.copy()
    elif k == 27:
        break
#add values to results dictionary
results["count"] = count
results["area"] = area
results["perimeter"] = perimeter
results["center"] = center
results["d_eq"] = d_eq
results["a_semimajor"] = a_semimajor
results["b_semiminor"] = b_semiminor
results["angle_ver"] = angle_ver
results["eccentricity"] = eccentricity
results["ellipses_list"] = ellipses_list
results["AJSC"] = mean(jaccard_score_list)
 
# Create the Pandas DataFrame
df = pd.DataFrame(results)
print()
print(df)
print()
 
# Export the dataframe to a csv file
df.to_csv(path_or_buf = output_data, index = None, header=True) 

# save annotated image
print(output_image)
cv2.imwrite(output_image, image)
 
# Destroy all generated windows:
cv2.destroyAllWindows()

