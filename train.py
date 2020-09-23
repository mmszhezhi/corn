import numpy as np
import sys, os, glob
from util.imgpreprocessing import utils
import cv2
from keras.applications import vgg16
import pandas as pd
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.losses import MSE
from sklearn.metrics import mean_squared_error, r2_score
from tsfresh import extract_features
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout
from keras import Sequential
from keras import optimizers
import time
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3, 5, 6, 7, 8, 10],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10, 50, 100, 150, 200]
}
import shutil

import logging

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class LaiModel(utils):

    def __init__(self, batch_size=10):
        super(LaiModel, self).__init__()
        self.target_size = (600, 500)
        self.batch_size = 3
        self.train_batch_size = 20
        self.img_gen_batch_size = 3
        self.period = 1
        self.name2lai = {}
        self.name2bin = {}
        self.init_labels()
        # self.init_model()
        # self.model_initial_vgg()

    def init_model(self):
        est = GradientBoostingRegressor()
        self.model = GridSearchCV(estimator=est, param_grid=param_grid,
                                  cv=2, n_jobs=-1, verbose=2, scoring=["neg_mean_squared_error"],
                                  return_train_score=True, refit="neg_mean_squared_error")

    def train_gbr(self):
        self.model.fit()
        # self.model.

    def init_labels(self):
        lb = pd.read_csv("labels_bin.csv", index_col=0)
        for record in lb.to_records():
            self.name2lai.update({record[1]: round(record[2], 3)})
            self.name2bin.update({record[1]: round(record[3], 3)})

    def batch_gen(self):
        assert self.length > 1
        for i in range((self.length // self.batch_size) + 1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            batch = self.total[start:min(end, self.length)]
            temp = []
            tname = []
            try:
                for img in batch:
                    origin = cv2.imread(img)
                    origin = cv2.resize(origin, (800, 600))
                    scaled = self.green_scaling(origin)  # ->np.array
                    scaled_expand = scaled[np.newaxis, :, :, :]
                    temp.append(scaled_expand)
                    tname.append(img.split("\\")[-1])
                yield tname, np.concatenate(temp, )
            except Exception as e:
                print(repr(e))

    def model_initial_vgg(self):
        model = Sequential()
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(600, 500, 3))
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

    def steps_counter(self, dsrc):
        dsrc = dsrc if isinstance(dsrc, list) else glob.glob(f"{dsrc}/*")
        assert isinstance(dsrc, list) and len(dsrc) > 0, "empty diretory error"
        length = len(dsrc)
        steps = math.ceil(length // self.batch_size)
        return steps

    def img_gen(self, dsrc, batch_size, epochs=-1,resize=(500,600), img_no=False,raw_img=False):
        """
        generate batch of images constantly
        :param epochs: -1 infinit iteration
        :param batch_size:
        :param dsrc: list of image file or directory of images
        :return:
        """
        dsrc = dsrc if isinstance(dsrc, list) else glob.glob(f"{dsrc}/*")
        assert isinstance(dsrc, list) and len(dsrc) > 0, "empty diretory error"
        length = len(dsrc)
        steps = math.ceil(length // batch_size)
        while epochs != 0:
            epochs -= 1
            logging.debug(f"epoch count {epochs if epochs // 1 == 0 else 'infinit'}")
            for i in range(steps):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch = dsrc[start:min(end, length)]
                temp = []
                tname = []
                tlabels = []
                try:
                    for img in batch:
                        origin = cv2.imread(img)
                        if raw_img:
                            origin = cv2.resize(origin, resize)
                            origin = self.green_scaling(origin)  # ->np.array
                        scaled_expand = origin[np.newaxis, :, :, :]
                        temp.append(scaled_expand)
                        iname = img.split("\\")[-1].split("-")[0]
                        tlabels.append(self.name2lai[iname])
                        tname.append(iname)
                    if img_no:
                        yield np.concatenate(temp), np.array(tlabels).reshape([-1, 1]), tname
                    else:
                        yield np.concatenate(temp), np.array(tlabels).reshape([-1, 1])
                except Exception as e:
                    print(repr(e))

    def train_vgg(self, epochs, train_dir, val_dir, store_dir="ws", period=1):
        """
        train a regerssion model base on vgg16
        :return:
        """
        train_gen = self.img_gen(train_dir, self.batch_size)
        val_gen = self.img_gen(val_dir, self.batch_size)
        t1 = time.time()
        filepath = store_dir + "/{epoch:02d}-{val_mean_squared_error:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, period=period, monitor='val_mean_squared_error', verbose=1,
                                     save_best_only=False,
                                     mode='max')
        callbacks_list = [checkpoint]
        print(f"training start \n store dir {filepath}")
        self.model1.fit_generator(train_gen, self.steps_counter(train_dir), epochs, validation_data=val_gen,
                                  validation_steps=self.steps_counter(val_dir), callbacks=callbacks_list)
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

    def evaluate(self, w, test_set, batch_size=3):
        """
        evaluate given model on give test set
        :param test_set: give test set list of image file or image directory
        :param w: given model weight
        :param threshold:
        :return:
        """
        model = self.load_model("m1", f"ws/{w}.h5")  # type Sequential
        val_gen = self.img_gen(test_set, batch_size, 1, img_no=True)
        l, p, no = [], [], []
        for data, labes, nos in val_gen:
            predicts = model.predict(data)
            p.extend([x[0] for x in predicts])
            l.extend([x[0] for x in labes])
            no.extend(nos)
        df = pd.DataFrame({"img_no": no, "label": l, "predict": p})
        mse = tf.keras.losses.mean_squared_error(df["label"], df["predict"])
        subd = str.split(test_set, '\\')[-1]
        subd = str.split(subd, '/')[-1]
        df.to_csv(f"{w.split('-')[-1]}-{mse.numpy()}-{subd}.csv")

    def evaluate_multi_dir(self, model, dir):
        for subdir in glob.glob(dir + "/*"):
            if "subtest" in subdir:
                logging.info(f"evaluation {subdir} on processing")
                self.evaluate(model, subdir)

    def encode(self):
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False)

        targets, datas = [], []
        i = 0
        for names, batch in self.batch_gen():
            vectors = vgg_conv.predict_on_batch(batch)
            targets.extend(names)
            datas.append(np.sum(vectors, axis=3))
            print(f"{i} of {self.length / self.batch_size}")
            i += 1

        df = pd.DataFrame(targets, columns=["id"])
        df.to_csv("labels2.csv")
        s = np.concatenate(datas, axis=0)
        np.save("encode2", s)

    def name2label(self, names):
        ret = []
        for name in names:
            ret.append(self.name2lai[name.split("-")[0]])
        return ret

    def feature_transform(self):
        t = []
        for data, labels in self.train_gen():
            for r in range(data.shape[0]):
                features = extract_features(pd.DataFrame({"id": [0] * data.shape[1], "value": data[r]}), column_id="id")
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

    def flatten_batch(self, batch: np.ndarray):
        t = []
        for index in range(batch.shape[0]):
            t.append(batch[index].flatten())
        return np.stack(t)

    def train_gen(self, dsrc):
        data = np.load(dsrc)  # type:np.ndarray
        labels = pd.read_csv("labels.csv")  # pd.DataFrame
        labels = labels["id"]
        for i in range((labels.shape[0] // self.train_batch_size + 1)):
            start = i * self.train_batch_size
            end = min((i + 1) * self.train_batch_size, labels.shape[0])
            batch = data[start:end]
            batch_l = labels[start:end]
            yield self.flatten_batch(batch), self.name2label(batch_l)

    def separate2bins(self, green_scale=False):
        dst = f"../imgandlai/bins2/"
        os.makedirs(dst, exist_ok=True)
        for img in glob.glob(f"../imgandlai/augdata4/*"):
            class_name = self.name2bin[img.split("-")[0].split("\\")[-1]]
            os.makedirs(dst + str(class_name), exist_ok=True)
            if green_scale:
                raw = cv2.imread(img)
                origin = cv2.resize(raw, (500, 600))

                origin = self.green_scaling(origin, trans=False)
                cv2.imwrite(dst + str(class_name) + "/" + img.split("\\")[-1], origin * 255)
            else:
                shutil.copy(img, dst + str(class_name))

    def gen_bins(self, path="../imgandlai/bins2"):
        generator = ImageDataGenerator(
            rescale=1. / 255, validation_split=0.2, )
        # for file in files:
        train_gen = generator.flow_from_directory(path, subset="training", target_size=self.target_size, shuffle=True,
                                                  batch_size=self.batch_size, )
        val_gen = generator.flow_from_directory(path, subset="validation", target_size=self.target_size, shuffle=True,
                                                batch_size=self.batch_size, )
        return train_gen, val_gen

    def val_gen_bins(self, path="../imgandlai/bins2"):
        generator = ImageDataGenerator(
            rescale=1. / 255)
        gen = generator.flow_from_directory(path, target_size=self.target_size, shuffle=True,
                                            batch_size=self.batch_size, )
        return gen

    def build_vgg_clf(self):
        model = Sequential()
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(600, 500, 3))
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False
        model.add(vgg_conv)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['accuracy'])
        self.model = model
        return model

    def train_bins(self, store_dir, epochs=10, ):
        train_gen, val_gen = self.gen_bins()
        label_map = "\n".join(train_gen.class_indices.keys())
        with open("label_map.txt", "w") as f:
            f.write(label_map)
        self.model = self.build_vgg_clf()
        os.makedirs(store_dir, exist_ok=True)
        filepath = store_dir + "/{epoch:02d}-{val_accuracy:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, period=self.period, monitor='val_accuracy', verbose=1,
                                     save_best_only=False,
                                     mode='max')
        callbacks_list = [checkpoint]
        t1 = time.time()
        print(f"training start \n store dir {filepath}")
        train_gen.image_shape
        self.model.fit_generator(train_gen, steps_per_epoch=train_gen.n // train_gen.batch_size, epochs=epochs,
                                 validation_data=val_gen,
                                 validation_steps=val_gen.n // val_gen.batch_size, callbacks=callbacks_list)
        # self.model.fit_generator(train_gen,steps_per_epoch=3,epochs=epochs,validation_data=val_gen,
        #                           validation_steps=3, callbacks=callbacks_list)
        t2 = time.time()
        print(f"process time {t1}-{t2}, total {t2 - t1}")
        model_json = self.model.to_json()
        with open("m2", "w") as jsonfile:
            jsonfile.write(model_json)
        self.model.save_weights("w2.h5")

    def load_bins_model(self, m, weights):
        model_json = open(m, "r")
        model = model_from_json(model_json.read())
        model_json.close()
        model.load_weights(weights)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['accuracy'])
        return model

    def evalue_bins_model(self):
        self.model = self.load_bins_model("m2", "w2.h5")
        gen = self.val_gen_bins("../imgandlai/bins2")
        with open("label_map.txt", "r") as f:
            lables = f.read().split("\n")
        ground_truth = list(gen.class_indices.keys())
        ls, ps = [], []
        steps = gen.n // gen.batch_size
        for b in range(gen.n // gen.batch_size):
            print(f"step {b} of {steps}")
            data, label = next(gen)
            predicts = self.model.predict(data)
            for i in range(predicts.shape[0]):
                prd = lables[np.argmax(predicts[i])]
                lb = ground_truth[np.argmax(label[i])]
                ls.append(lb)
                ps.append(prd)
            if b > 5:
                break
        df = pd.DataFrame({"label": ls, "predict": ps})
        mse = mean_squared_error(df["label"], df["predict"])
        r2 = r2_score(df["label"], df["predict"])
        # mse = tf.keras.losses.mean_squared_error(df["label"], df["predict"])
        df.to_csv(f"summary-{mse}-{r2}.csv")

    def evalue_bins_regre(self):
        gen = self.img_gen("../imgandlai/test2",3,1,img_no=True,raw_img=True)
        self.model = self.load_bins_model("m2", "w2.h5")
        with open("label_map.txt", "r") as f:
            lables = f.read().split("\n")
        ls, ps = [], []
        for data,l,name in gen:
            predicts = self.model.predict(data)
            for i in range(predicts.shape[0]):
                prd = lables[np.argmax(predicts[i])]
                lb = self.name2lai[name[i]]
                ls.append(lb)
                ps.append(prd)
            # break
        df = pd.DataFrame({"label": ls, "predict": ps})
        mse = mean_squared_error(df["label"], df["predict"])
        r2 = r2_score(df["label"], df["predict"])
        # mse = tf.keras.losses.mean_squared_error(df["label"], df["predict"])
        df.to_csv(f"summary-{round(mse,2)}-{round(r2,2)}-regre.csv")



df = pd.DataFrame({"data": np.random.normal(1, 2, 100), "id": [1] * 50 + [2] * 50})
if __name__ == '__main__':
    model = LaiModel()
    # model.train_bins("ws 3",epochs=20)
    model.evalue_bins_regre()
    # model.separate2bins(green_scale=True)
    # model.train_vgg(12,"../imgandlai/augdata3","../imgandlai/test")
    # model.evaluate("12-0.11","../imgandlai/test0")
    # model.evaluate_multi_dir("12-0.11","../imgandlai")
    # model.encode()
    # gen = model.img_gen()
    # a,b = next(gen)
    # print(b)
    # gen = model.train_gen()
    # model.feature_transform()
    # print(arr)
