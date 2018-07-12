from baseline_aug import get_unet, input_shape
import glob
from cv2 import imread
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import label
from pycocotools import mask as maskUtils
from tqdm import tqdm
import os
import cv2
from keras.layers import ReLU

batchsize = 64

def batch(iterable, n=batchsize):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == '__main__':
    model_name = "baseline_unet_aug_do_0.1_activation_ReLU_"

    #test_files = glob.glob("../input/validation_input/*.jpg")
    test_files = glob.glob("../input/test_input/*.jpg")
    try:
        os.mkdir("../output/"+model_name)
    except:
        pass

    model = get_unet(do=0.1, activation=ReLU)

    file_path = model_name + "weights.best.hdf5"

    model.load_weights(file_path)

    for batch_files in tqdm(batch(test_files), total=len(test_files)//batchsize):

        imgs = [resize(imread(image_path)/255., input_shape) for image_path in batch_files]

        imgs = np.array(imgs)


        pred = model.predict(imgs)

        pred_all = (pred)

        pred = np.clip(pred, 0, 1)

        for i, image_path in enumerate(batch_files):

            pred_ = pred[i, :, :, 0]
            pred_ = resize(pred_, (400, 275))

            pred_ = 255.*(pred_ - np.min(pred_))/(np.max(pred_)-np.min(pred_))

            print(np.max(pred_), np.min(pred_))
            image_base = image_path.split("/")[-1]

            cv2.imwrite("../output/"+model_name+"/"+image_base, pred_, [int(cv2.IMWRITE_JPEG_QUALITY), 50])





