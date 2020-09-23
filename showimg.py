import cv2
import numpy as np
data = np.load("origin.npy")
print(data.shape)
cv2.imshow("de.jpg",data)
cv2.waitKey(0)
