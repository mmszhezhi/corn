import glob
import shutil
import os
dst = f"../imgandlai/test2"
os.makedirs(dst,exist_ok=True)
for f in glob.glob(f"../imgandlai/bins2/*"):
        for img in glob.glob(f"{f}/*"):
            if "0" == img.split("-")[-1].split(".")[0]:
                print(img)
                shutil.copy(img,dst)