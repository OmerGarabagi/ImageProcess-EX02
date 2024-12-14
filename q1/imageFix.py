# Omer Garabagi, 322471145
# Omer Chernia, 318678620

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_fix(image, id):
    if id == 1:
        # Apply histogram equalization for Image 1
        return cv2.equalizeHist(image)
    elif id == 2:
        # Apply gamma correction for Image 2
        gamma = 2.2
        inv_gamma = 1.0 / gamma
        table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8"))
        return cv2.LUT(image, table)
    elif id == 3:
        # change nothing for Image 3, return the original image because it looks good as it is
        return image
	

for i in range(1,4):
	path = f'{i}.jpg'
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	fixed_image = apply_fix(image, i)
	plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)
