import glob,os,sys
import shutil

img_dir = "../imgandlai/img"
label = "imgandlai/lai"
dest = "../imgandlai/source"
os.makedirs(dest,exist_ok=True)
for f in glob.glob(img_dir + "/*"):
    for f2 in glob.glob(f+"/*"):
        for img in glob.glob(f2 + "/*"):
            print(img)
            shutil.copy(img,dest)



