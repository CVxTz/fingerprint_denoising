# part of this script was taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation
from glob import glob
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, PReLU, ReLU
from keras.models import Model
from keras.activations import relu
from keras.optimizers import Adam
from numpy import random
from sklearn.model_selection import KFold
from skimage.transform import resize
from sklearn.model_selection import train_test_split

input_shape = (400, 288)


def custom_activation(x):
    return K.relu(x, alpha=0.0, max_value=1)

smooth = 1.



def get_unet(do=0, activation=ReLU):
    inputs = Input(input_shape+(3,))
    conv1 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(conv1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv5)))

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(conv6)))

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(conv7)))

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(conv8)))

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(conv9)))

    conv10 = Dropout(do)(Conv2D(1, (1, 1), activation='sigmoid')(conv9))

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=losses.mse)

    model.summary()

    return model




#masks_tr_tr = masks_tr_tr[... ,np.newaxis]
batch_size = 8
from aug_utils import random_augmentation
import cv2


def read_input(path):
    x = resize(cv2.imread(path)/255., input_shape)
    return np.asarray(x)

def read_gt(path):
    y = resize(cv2.imread(path, 0)/255., input_shape)
    return np.asarray(y)[..., np.newaxis]


def gen(data):
    while True:
        # choose random index in features
        # try:
        index= random.choice(list(range(len(data))), batch_size)
        index = list(map(int, index))
        list_images_base = [read_input(data[i][0]) for i in index]
        list_gt_base = [read_gt(data[i][1]) for i in index]


        list_images_aug = []
        list_gt_aug = []

        for image_, gt in zip(list_images_base, list_gt_base):
            image_aug, gt = random_augmentation(image_, gt) #image_, gt

            list_images_aug.append(image_aug)
            list_gt_aug.append(gt)

        yield np.array(list_images_aug), np.array(list_gt_aug)
        # except Exception as e:
        #     print(e)





if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dropout", required=False,
                    help="dropout", type=float, default=0)
    ap.add_argument("-a", "--activation", required=False,
                    help="activation", default="ReLu")

    args = vars(ap.parse_args())

    # if "dropout" not in args:
    #     args['dropout'] = 0
    #
    # if "activation" not in args:
    #     args['activation'] = "ReLu"

    activation = globals()[args['activation']]

    model_name = "baseline_unet_aug_do_%s_activation_%s_"%(args['dropout'], args['activation'])

    print("Model : %s"%model_name)

    train_data = list(zip(sorted(glob('../input/training_input/*.jpg')), sorted(glob('../input/training_ground-truth/*.jpg'))))
    val_data = list(zip(sorted(glob('../input/validation_input/*.jpg')), sorted(glob('../input/validation_ground-truth/*.jpg'))))

    print(len(val_data)//batch_size, len(val_data), batch_size)


    model = get_unet(do=args['dropout'], activation=activation)

    file_path = model_name + "weights.best.hdf5"
    try:
        model.load_weights(file_path, by_name=True)
    except:
        pass


    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=1)
    callbacks_list = [checkpoint, early, redonplat]  # early

    history = model.fit_generator(gen(train_data), validation_data=gen(val_data), epochs=1000, verbose=2,
                         callbacks=callbacks_list, steps_per_epoch= len(train_data)//batch_size,
                                  validation_steps=len(val_data)//batch_size, use_multiprocessing=False, workers=16)




