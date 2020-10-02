import os,glob,cv2
from util import imgpreprocessing
import numpy as np



if __name__ == '__main__':
    scale = 2
    src = "../imgandlai/source"
    dst = "../imgandlai/p2/"
    os.makedirs(dst,exist_ok=True)
    for img in glob.glob(src+'/*'):
        origin = cv2.imread(img)
        origin = cv2.resize(origin,(500,600))
        # origin.astype(np.uint8)
        origin = imgpreprocessing.utils.green_scaling(origin,True,1)
        oringin = origin[:,:,np.newaxis]
        # origin.astype(np.float64)
        # gray = cv2.cvtColor(origin,cv2.COLOR_BGR2BGRA)
        name = os.path.basename(img)
        cv2.imwrite(dst+name,origin*255)
        # break
