import glob,os,sys
import shutil

img_dir = "imgandlai/img"
label = "imgandlai/lai"
dest = "imgandlai/source"
for f in glob.glob(img_dir + "/*"):
    for f2 in glob.glob(f+"/*"):
        for img in glob.glob(f2 + "/*"):
            shutil.copy(img,dest)



