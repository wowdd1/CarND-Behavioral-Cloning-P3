import pandas as pd
import cv2
import random
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, ELU, Lambda, merge, MaxPooling2D, Input, Activation, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.callbacks import Callback
from keras.optimizers import Adam
import numpy as np
from sklearn.utils import shuffle
import json
import preprocess_util as preprocess
import keras.backend as K
import gc

def crop_top_and_bottom(image):
    resized = cv2.resize(image[70:140], (64,64),  cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)[:,:,1]


def shift_img(image, random_shift):
    rows, cols = image.shape
    mat = np.float32([[1, 0, random_shift], [0, 1, 0]])
    return cv2.warpAffine(image, mat, (cols, rows))

def load_image(row):
    fileName = "./data/{0}".format(row.image.strip())
    #print(fileName)
    image = cv2.imread(fileName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_top_and_bottom(image)

    if(row.is_flipped):
        image = cv2.flip(image,1)
    if(row.is_shift):
        image = shift_img(image, row.random_shift)
    return image

def get_processed_dataframes():

    return pd.read_csv('preprocessed_driver_log.csv')

#hyperparameters
INPUT_SHAPE = (64, 64, 1)
LEARNING_RATE = 1e-1
BATCH_SIZE = 128
EPOCHS = 50

class CustomEarlyStop(Callback):

    def __init__(self, monitor='val_loss'):
        super(CustomEarlyStop, self).__init__()
        self.monitor = monitor


    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss <= 0.039:
            print("\nEarly Stop on Epoch {0} with Val_loss {1}\n".format(epoch,val_loss))
            self.model.stop_training = True

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    c_axis = 3
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    elu = "elu_"

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('elu', name=s_id + elu + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('elu', name=s_id + elu + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('elu', name=s_id + elu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x

def squeeze_model_52():
    input_shape=(64, 64, 1)
    input_img = Input(shape=input_shape)
    x = Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape)(input_img)

    x = Convolution2D(2, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(x)
    x = Activation('elu', name='elu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=1, expand=2)
    x = Dropout(0.2, name='drop3')(x)


    x = GlobalAveragePooling2D()(x)
    out = Dense(1, name='loss')(x)
    model = Model(input=input_img, output=[out])

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
    return model


def load_all_images():
    driverlog_df = get_processed_dataframes()
    print("driverlog_df loaded: ", len(driverlog_df))
    print("Loading images from log file now....")
    features = [load_image(row) for _, row in driverlog_df.iterrows()]
    labels   = driverlog_df.steering

    return np.array(features).reshape(len(features), 64, 64, 1), labels

def train():
    model = squeeze_model_52()
    
    model.summary()

    features, labels = load_all_images()
    #this is the early termination callback. Stops trainig when val_loss <=.039
    early_stop = CustomEarlyStop(monitor='val_loss')
    model.fit(x=features,
              y=labels,
              verbose=1,
              batch_size=BATCH_SIZE,
              nb_epoch=EPOCHS,
              validation_split=0.3,
              callbacks=[early_stop])

    model.save("model.h5")
    print("Training complete!")

    #Clean up after ourselves!!
    K.clear_session()
    gc.collect()

train()