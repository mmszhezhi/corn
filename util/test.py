from util.imgpreprocessing import utils
util = utils()
import glob
import cv2
import matplotlib.pyplot as plt


data = {}
for imgdir in glob.glob("imgs/*"):
    img = cv2.imread(imgdir)
    img = cv2.resize(img,(800,600))
    scaled = util.green_scaling(img)
    data.update({imgdir.split('/')[-1]:scaled})


fig = plt.figure(figsize=(60,50))
for i,(k,v) in enumerate(data.items()):
    ax = plt.subplot(6,1,i+1)
    plt.imshow(v)
    plt.title(k)
plt.show()
# plt.savefig("green_scaled.jpg")