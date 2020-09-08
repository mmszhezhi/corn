import glob
import shutil
dst = f"../imgandlai/test"
for img in glob.glob(f"../imgandlai/augdata/*"):
    if "0" == img.split("-")[-1].split(".")[0]:
        shutil.move(img,dst)