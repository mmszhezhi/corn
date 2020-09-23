import glob
import shutil
import os
import pandas as pd
import numpy as np
lb = pd.read_excel("labels.xls")
label = []
def train(x):
    label.append(x["bins"].mid)
lb["bins"] = pd.cut(lb["LAI"],30)
lb.apply(train,axis=1)
lb["label"] = label
lb.drop(["bins"],axis=1,inplace=True)
lb.to_csv("labels_bin.csv")
print(lb)