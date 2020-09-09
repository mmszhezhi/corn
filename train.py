import numpy as np
import sys,os,glob
from util.imgpreprocessing import utils
import cv2
from keras.applications import vgg16
import pandas as pd
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tsfresh import extract_features
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense,Activation,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,Flatten,Dropout
from keras import Sequential
from keras import optimizers
import time
import math
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3,5,6,7,8,10],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10,50,100,150,200]
}

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
class LaiModel(utils):

    def __init__(self,batch_size=10,src="../imgandlai/augdata",test_set="../imgandlai/test"):
        super(LaiModel,self).__init__()
        self.src = src
        self.test_src = test_set
        self.batch_size = 3
        self.train_batch_size = 20
        self.total = glob.glob(f"{self.src}/*")
        self.test_set = glob.glob(f"{self.test_src}/*")
        self.img_gen_batch_size = 3

        self.length = len(self.total)
        self.val_length = len(self.test_set)

        self.steps = (self.length // self.img_gen_batch_size) +1
        self.val_steps = (self.val_length // self.img_gen_batch_size)+1
        self.name2lai = {}
        self.init_labels()
        # self.init_model()
        self.model_initial_vgg()

    def init_model(self):
        est = GradientBoostingRegressor()
        self.model = GridSearchCV(estimator=est, param_grid=param_grid,
                                   cv=2, n_jobs=-1, verbose=2, scoring=["neg_mean_squared_error"],
                                   return_train_score=True, refit="neg_mean_squared_error")

    def train_gbr(self):
        self.model.fit()
        # self.model.

    def init_labels(self):
        lb = pd.read_excel("labels.xls")
        for record in lb.to_records():
            self.name2lai.update({record[1]:record[2]})

    def batch_gen(self):
        assert self.length >1
        for i in range((self.length // self.batch_size) + 1):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            batch = self.total[start:min(end,self.length)]
            temp = []
            tname = []
            try:
                for img in batch:
                    origin = cv2.imread(img)
                    origin = cv2.resize(origin, (800, 600))
                    scaled = self.green_scaling(origin) #->np.array
                    scaled_expand = scaled[np.newaxis,:,:,:]
                    temp.append(scaled_expand)
                    tname.append(img.split("\\")[-1])
                yield tname, np.concatenate(temp, )
            except Exception as e:
                print(repr(e))

    def model_initial_vgg(self):
        model = Sequential()
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(600,500,3))
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False
        model.add(vgg_conv)
        model.add(Flatten())
        model.add(Dense(312, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['mean_squared_error'])
        self.model1 = model
        return model

    def img_gen(self,dsrc,batch_size,epochs=-1):
        """
        generate batch of images constantly
        :param epochs: -1 infinit iteration
        :param batch_size:
        :param dsrc: list of image file or directory of images
        :return:
        """
        dsrc = dsrc if isinstance(dsrc,list) else glob.glob(f"{dsrc}/*")
        assert isinstance(dsrc,list) and len(dsrc) >0 ,"empty diretory error"
        length = len(dsrc)
        steps = math.ceil(length//batch_size)
        while epochs!=0:
            epochs-=1
            logging.debug(f"epoch count {epochs if epochs // 1 ==0 else 'infinit'}")
            for i in range(steps):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch = dsrc[start:min(end, length)]
                temp = []
                tname = []
                try:
                    for img in batch:
                        origin = cv2.imread(img)
                        origin = cv2.resize(origin, (600, 500))
                        scaled = self.green_scaling(origin)  # ->np.array
                        scaled_expand = scaled[np.newaxis, :, :, :]
                        temp.append(scaled_expand)
                        tname.append(self.name2lai[img.split("\\")[-1].split("-")[0]])
                    yield np.concatenate(temp,),tname
                except Exception as e:
                    print(repr(e))

    def train_vgg(self,epochs,store_dir="ws",period=1):
        """
        train a regerssion model base on vgg16
        :return:
        """
        train_gen = self.img_gen(self.total,self.batch_size)
        val_gen = self.img_gen(self.test_set,self.batch_size)
        t1 = time.time()
        filepath = "{}/weights-improvement-{epoch:02d}-{val_mean_squared_error:.2f}.h5".format(store_dir)
        checkpoint = ModelCheckpoint(filepath, period=period, monitor='val_mean_squared_error', verbose=1, save_best_only=False,
                                     mode='max')
        callbacks_list = [checkpoint]
        print(f"training start \n store dir {filepath}")
        self.model1.fit_generator(train_gen,self.steps,epochs,validation_data=val_gen,validation_steps=self.val_steps,callbacks=callbacks_list)
        t2 = time.time()
        print(f"process time {t1}-{t2}, total {t2 - t1}")
        model_json = self.model1.to_json()
        with open("m1", "w") as jsonfile:
            jsonfile.write(model_json)
        self.model1.save_weights("w1.h5")

    def load_model(self, m, weights):
        model_json = open(m, "r")
        model = model_from_json(model_json.read())
        model_json.close()
        model.load_weights(weights)
        model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    def evaluate(self,w,test_set,batch_size=3):
        """
        evaluate given model on give test set
        :param test_set: give test set list of image file or image directory
        :param w: given model weight
        :param threshold:
        :return:
        """
        model = self.load_model("m1",f"ws/{w}.h5")  # type Sequential
        val_gen = self.img_gen(test_set,batch_size,1)
        l,p = [],[]
        for data,labes in val_gen:
            predicts = model.predict(data)
            print(predicts)
            p.extend([x[0] for x in predicts])
            l.extend(labes)
        df = pd.DataFrame({"label":l,"predict":p})
        mse = np.sum((df["label"] - df["predict"])**2)/21
        df.to_csv(f"{w}-{mse}-predict.csv")

    def encode(self):
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False)

        targets,datas=[],[]
        i = 0
        for names,batch in self.batch_gen():
            vectors = vgg_conv.predict_on_batch(batch)
            targets.extend(names)
            datas.append(np.sum(vectors,axis=3))
            print(f"{i} of {self.length / self.batch_size}")
            i+=1

        df = pd.DataFrame(targets,columns=["id"])
        df.to_csv("labels2.csv")
        s = np.concatenate(datas,axis=0)
        np.save("encode2",s)

    def name2label(self,names):
        ret = []
        for name in names:
            ret.append(self.name2lai[name.split("-")[0]])
        return ret

    def feature_transform(self):
        t = []
        for data,labels in self.train_gen():
            for r in range(data.shape[0]):
                features = extract_features(pd.DataFrame({"id":[0]*data.shape[1],"value":data[r]}), column_id="id")
                t.append(features)
        return np.save(np.stack(t))

    def load_model(self, m, weights):
        model_json = open(m, "r")
        model = model_from_json(model_json.read())
        model_json.close()
        model.load_weights(weights)
        model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def flatten_batch(self,batch:np.ndarray):
        t = []
        for index in range(batch.shape[0]):
             t.append(batch[index].flatten())
        return np.stack(t)

    def train_gen(self,dsrc):
        data = np.load(dsrc)#type:np.ndarray
        labels = pd.read_csv("labels.csv") #pd.DataFrame
        labels = labels["id"]
        for i in range((labels.shape[0] // self.train_batch_size + 1)):
            start = i*self.train_batch_size
            end = min((i+1)*self.train_batch_size,labels.shape[0])
            batch = data[start:end]
            batch_l = labels[start:end]
            yield self.flatten_batch(batch),self.name2label(batch_l)






df = pd.DataFrame({"data":np.random.normal(1,2,100),"id":[1]*50 + [2]*50})
if __name__ == '__main__':
    model = LaiModel()
    # model.train_vgg()
    model.evaluate("weights-improvement-12-0.05",model.test_set)
    # model.encode()
    # gen = model.img_gen()
    # a,b = next(gen)
    # print(b)
    # gen = model.train_gen()
    # model.feature_transform()
    # print(arr)















