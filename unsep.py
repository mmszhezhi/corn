import glob
import shutil
import os
src = f"../imgandlai/bins2"
dst = "../imgandlai/val/"
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f"{src}/*"):
    for img in glob.glob(f"{f}/*"):
        if "0" in img.split('/')[-1]:
            shutil.copy(img,dst+img.split('\\')[-1])