import glob
import shutil
import os
dst = f"../imgandlai/test0"
os.makedirs(dst,exist_ok=True)
for img in glob.glob(f"../imgandlai/augdata3/*"):
    if "0" == img.split("-")[-1].split(".")[0]:
        shutil.copy(img,dst)