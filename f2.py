import numpy as np
from tsfresh import extract_features
# data = np.load("encode1.npy")
# print(data.shape)
import joblib

import pandas as pd
import numpy as np




df = pd.DataFrame({"data":np.random.normal(1,2,100),"id":[1]*50 + [2]*50})
if __name__ == '__main__':
    data = np.load("encode1.npy")
    data[0].flatten()
    arr = extract_features(df,column_id="id")
    print(arr)

