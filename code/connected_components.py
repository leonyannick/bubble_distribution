import cv2
import numpy as cp
import matplotlib.pyplot as plt 
import matplotlib as mpl

def read_image(input_image):
    original = cv2.imread(input_image)
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    return original, image

org, img = read_image("graphics/opening/connected_components_opening_783.png")
cv2.imshow("img", img)
cv2.waitKey(0)

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

print(retval)

print(stats)

mpl.rcParams.update({'font.size': 15})

pixel_count = [x[4] for x in stats]
print(pixel_count)

fig, ax = plt.subplots(figsize=(5, 5))

bins = range(0,100)
n, bins, patches = ax.hist(pixel_count, bins)

ax.set_xlabel("number of pixels")
ax.set_ylabel("count")
ax.set_ylim([0,30])


plt.tight_layout()
plt.show()

name = "graphics/opening/pixels hist_opening"
fig.savefig(f"{name}.pdf")