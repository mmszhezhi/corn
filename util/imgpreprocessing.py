import cv2
import numpy as np
import matplotlib.pyplot as plt



class utils:
    def construct_img(self,f,s,t):
        img = np.zeros([1000,1000,3],dtype=np.uint8)
        img[:,:,0] += f
        img[:, :,1 ] += s
        img[:, :, 2] += t
        return img

    @staticmethod
    def green_scaling(img:np.ndarray,channel,trans=False):
        # assert img.dtype == np.uint8
        if trans and img.shape[0] > img.shape[1]:
            img = img.transpose(1,0,2)
        sumation = img.sum(axis=2) + 0.00001
        ratio_02 = np.stack([img[:,:,0] / sumation,img[:,:,2]/sumation],axis=2)
        # ratio_02[ratio_02 >100] = 1
        green_channel = img[:, :, 1]
        green_channel[green_channel > 210] = 1000
        ratio_1 = green_channel / sumation
        sum_ratio_02 = ratio_02.sum(axis=2)
        f_channel = 0.08*img[:, :, 0] /sumation
        l_channel = 0.08*img[:, :, 2] /sumation
        im = np.clip(ratio_1 / (sum_ratio_02 + 0.00001) / 3 , 0, 1)
        im[im < 0.195] = 0
        if channel == 1:
            return im
        elif channel ==3:
            return np.stack((f_channel,im,l_channel),axis=2)

    def adjust_img(self,img):
        ad = img[:,:,1]
        ad[ad < 210] = 0
        plt.imshow(ad)
        plt.show()


import glob

util = utils()
for img in glob.glob("imgs/*"):
    if img.split('\\')[-1] != "F1683-0.jpg":
        continue
    origin = cv2.resize(cv2.imread(img),(600,500))
    im = util.green_scaling(origin)
    cv2.imwrite("test.jpg",im*255)
    plt.imshow(im)
    ratio = np.sum(im > 0.1) / np.multiply(im.shape[0],im.shape[1])
    plt.title(ratio)
    plt.show()
    break

#
# img = util.construct_img(0,180,0)
# # # cv2.imshow("jpg",img)
# plt.imshow(img)
# plt.show()

# util.adjust_img(origin)

# im = util.green_scaling(origin)
# # im = np.clip(green_scaled /3,0,1)
# # im[im < 0.195] = 0

# print(ratio)
#

