import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import ndimage as ndi
import matplotlib as mpl


#https://stackoverflow.com/questions/31805560/how-to-create-surface-plot-from-greyscale-image-with-matplotlib

image = np.zeros((500, 500), dtype="uint8")

circle1 = cv2.circle(image, (250,170), 100, 255, -1)
circle2 = cv2.circle(image, (250,320), 100, 255, -1)

dist = cv2.distanceTransform(image, cv2.DIST_L2, 3)
mpl.rcParams.update({'font.size': 15})

plt.imshow(image, cmap='Greys_r')
#plt.imshow(dist, cmap='Greys_r')
plt.xlabel("x")
plt.ylabel("y")
plt.waitforbuttonpress(0)




# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:dist.shape[0], 0:dist.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, dist ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.set_zlabel("pixel intensity")
ax.set_zlim(80,0)

# show it
plt.show()
