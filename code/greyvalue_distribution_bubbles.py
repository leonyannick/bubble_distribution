import cv2
import numpy as np
import matplotlib.pyplot as plt 


# Read image
im = cv2.imread("pictures/c_P1260783.png", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", im)

# Select ROI
fromCenter = False
r = cv2.selectROI("Image", im, fromCenter)
cv2.rectangle(im, (int(r[0]-10),int(r[1])-10), (int(r[0]+r[2]+10),int(r[1]+r[3])+10), (255,255,255), 10)
# Crop image
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Display cropped image
cv2.imshow("Image", imCrop)
#cv2.imwrite("pictures/single_bubble.png", imCrop)
cv2.waitKey(0)
imCrop = cv2.cvtColor(imCrop, cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
shape = imCrop.shape




#small change to display histogram
#bins = range(0,256)
height_median = int(shape[0] / 2)
grey_values = [x for x in imCrop[height_median]]
bins = range(shape[1])
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
ax = axes

#draw horizontal line
cv2.line(imCrop,(0,int(shape[0]/2)+1),(shape[1],int(shape[0]/2)+1),(0,0,0),1)


#ax[0].imshow(im)
#ax[0].set_title('image')
ax[0].imshow(imCrop)
#ax[1].set_title('bubble')
ax[0].set_xlabel('x', fontsize="14")
ax[0].set_ylabel('y', fontsize="14")

#plt.hist(imCrop.ravel(),256,[0,256]) 
#plt.show()

plt.plot(bins, grey_values, color='red', label='grey values')
ax[1].set_xlabel('x', fontsize="14")
ax[1].set_ylabel('Intensity', fontsize="14")
plt.tight_layout()
plt.show() 

fig.savefig('graphics/graylevel_distribution/real_bubble.pdf')
"""
cv2.imshow("Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""