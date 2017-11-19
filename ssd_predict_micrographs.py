from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from os import walk,listdir
from numpy import ceil

from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

from keras.models import load_model


class predictor:


    def __init__():
        self.prediction_path = '/u/misser11/InSilicoTem_Python_Version/SSD/predictions/'
        self.model = load_model('ssd300_0.h5')

    def __predictions(self):
        onlyfiles = [f for f in listdir(self.prediction_path) if isfile(join(self.prediction_path, f))]
        for img_path  in onlyfiles:
            img = mrcfile.open(img_path)
            img = img.data
            predictions_appender = _predict_image(self,img)
            _ploter(self,predictions_appender,onlyfiles)
            del predictions_appender
            del img


    def _predict_image(self,image):
        predictions_appender = []
        shaper = image.shape
        x_num_roll = ceil(shaper[0]/150)
        y_num_roll = ceil(shaper[1]/150)
        for x in range(x_num_roll):
            for y in range(y_num_roll):
                X = numpy.roll(image,150*x)
                X = numpy.roll(X,150*y)
                y_pred = model.predict(X[0:300,0:300])
                # 4: Decode the raw prediction `y_pred`
                y_pred_decoded = decode_y(y_pred,
                                          confidence_thresh=0.01,
                                          iou_threshold=0.45,
                                          top_k='all',
                                          input_coords='centroids',
                                          normalize_coords=normalize_coords,
                                          img_height=img_height,
                                          img_width=img_width)
                y_pred_decoded[:][2:3]+150*x
                y_pred_decoded[:][3:5]+150*y
                predictions_appender(y_pred_decoded[:])
        return np.array(predictions_appender)

    def _ploter(self,y_pred_decoded,name):
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print(y_pred_decoded)


        plt.figure(figsize=(20,12))
        current_axis = plt.gca()

        for box in y_pred_decoded:
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))
            current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

        for box in y_true[i]:
            label = '{}'.format(classes[int(box[0])])
            current_axis.add_patch(plt.Rectangle((box[1], box[3]), box[2]-box[1], box[4]-box[3], color='green', fill=False, linewidth=2))
            current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
        plt.save("%s.tiff", % name)
