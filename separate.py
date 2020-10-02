import glob
import shutil
import os
import numpy as np
import random
dst = f"../imgandlai/val"
os.makedirs(dst,exist_ok=True)
for img in glob.glob(f"../imgandlai/p2/*"):
    if random.randint(0,10)>8:
        print(img)
        shutil.copy(img,dst)