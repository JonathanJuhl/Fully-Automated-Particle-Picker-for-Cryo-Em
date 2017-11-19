from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from os import walk,listdir


from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator


### Set up the model
class model:
    def __init__():
        self.img_height = 300
        #path to directory. Directory must contain folders of class names and within each folder, images of that particular class.
        self.path_to_train_directory = '/u/misser11/InSilicoTem_Python_Version/SSD/Training_folder/'
        self.cvsfile = '/u/misser11/InSilicoTem_Python_Version/SSD/Fully-Automated-Particle-Picker-for-Cryo-Em/cvsfile.csv'
        self.path_to_validation_directory = '/u/misser11/InSilicoTem_Python_Version/SSD/validation/'
        self.cvsfile_val  = '/u/misser11/InSilicoTem_Python_Version/SSD/Fully-Automated-Particle-Picker-for-Cryo-Em/cvsfile_val.csv'
        self.img_width = 300
        self.epochs = 30
        self.img_train_height = 100
        self.img_train_width = 100
        self.img_channels = 1 # Number of color channels of the input images
        self.n_classes = len([x[0] for x in walk(self.path_to_train_directory)])+1 # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
        self.scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        self.aspect_ratios = [[0.5, 1.0, 2.0],
                         [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                         [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                         [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                         [0.5, 1.0, 2.0],
                         [0.5, 1.0, 2.0]] # The anchor box aspect ratios used in the original SSD300
        self.two_boxes_for_ar1 = True
        self.limit_boxes = True # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
        self.variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
        self.coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
        self.normalize_coords = True
        self.batch_size = 32

    def __load_model(self):
        # 2: Build the Keras model (and possibly load some trained weights)
        K.clear_session() # Clear previous models from memory.
        # The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
        model, predictor_sizes = ssd_300(image_size=(self.img_train_height, self.img_train_width, self.img_channels),
                                         n_classes=self.n_classes,
                                         min_scale=None, # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                         max_scale=None,
                                         scales=self.scales,
                                         aspect_ratios_global=None,
                                         aspect_ratios_per_layer=self.aspect_ratios,
                                         two_boxes_for_ar1=self.two_boxes_for_ar1,
                                         limit_boxes=self.limit_boxes,
                                         variances=self.variances,
                                         coords=self.coords,
                                         normalize_coords=self.normalize_coords)
        # Set the path to the VGG-16 weights below.

        ### Set up training


        # 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 4: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function

        ssd_box_encoder = SSDBoxEncoder(img_height=self.img_train_height,
                                        img_width=self.img_train_width,
                                        n_classes=self.n_classes,
                                        predictor_sizes=self.predictor_sizes,
                                        min_scale=None,
                                        max_scale=None,
                                        scales=self.scales,
                                        aspect_ratios_global=None,
                                        aspect_ratios_per_layer=self.aspect_ratios,
                                        two_boxes_for_ar1=self.two_boxes_for_ar1,
                                        limit_boxes=self.limit_boxes,
                                        variances=self.variances,
                                        pos_iou_threshold=0.5,
                                        neg_iou_threshold=0.2,
                                        coords=self.coords,
                                        normalize_coords=self.normalize_coords)
    def __load_training_data(self):
        train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

        train_classes = [x[0] for x in join(self.path_to_train_directory,walk(self.path_to_train_directory))+'/']

        train_labels = [x[0] for x in walk(self.path_to_train_directory)]

        train_dataset.parse_csv(images_path=train_classes,
                                labels_path=self.cvsfile,
                                include_classes='all',
                                ret=False)

        train_generator = train_dataset.generate(batch_size=batch_size,
                                                 train=True,
                                                 ssd_box_encoder=ssd_box_encoder,
                                                 equalize=False,
                                                 translate=False,
                                                 scale=False,
                                                 full_crop_and_resize=(self.img_train_height,self.img_train_width), # This one is important because the Pascal VOC images vary in size
                                                 random_crop=False,
                                                 crop=False,
                                                 resize=False,
                                                 gray=True,
                                                 limit_boxes=True, # While the anchor boxes are not being clipped, the ground truth boxes should be
                                                 include_thresh=0.4,
                                                 diagnostics=True)

        n_train_samples = train_dataset.get_n_samples() # Get the number of samples in the training dataset to compute the epoch length below
        return train_generator, n_train_samples
    def __load_validation_data(self):
             val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
            # 6: Create the validation set batch generator
             validation_classes = [x[0] for x in join(self.path_to_validation_directory,walk(self.path_to_validation_directory))+'/']
             val_dataset.parse_csv(images_path=validation_classes,
                                            labels_path=self.cvsfile_val,
                                            include_classes='all',
                                            ret=False)

             val_generator = val_dataset.generate(batch_size=self.batch_size,
                                                 train=True,
                                                 ssd_box_encoder=ssd_box_encoder,
                                                 equalize=False,
                                                 brightness=False,
                                                 flip=False,
                                                 translate=False,
                                                 scale=False,# This one is important because the Pascal VOC images vary in size
                                                 full_crop_and_resize=(self.img_train_height,self.img_train_width), # This one is important because the Pascal VOC images vary in size
                                                 random_crop=False,
                                                 crop=False,
                                                 resize=False,
                                                 gray=True,
                                                 limit_boxes=True,
                                                 include_thresh=0.4,
                                                 diagnostics=False)

             n_val_samples = val_dataset.get_n_samples()
             return val_generator, n_val_samples
            # 7: Define a simple learning rate schedule


            ### Run training

            # 7: Run training


    def __make_model(self):
        train_generator,n_train_samples = __load_training_data()
        val_generator,n_val_samples = __load_training_data()
        history = model.fit_generator(generator = train_generator,
                                      steps_per_epoch = ceil(n_train_samples/batch_size),
                                      epochs = self.epochs,
                                      callbacks = [ModelCheckpoint('./ssd300_weights_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                                                   monitor='val_loss',
                                                                   verbose=1,
                                                                   save_best_only=True,
                                                                   save_weights_only=True,
                                                                   mode='auto',
                                                                   period=1),
                                                   LearningRateScheduler(lr_schedule),
                                                   EarlyStopping(monitor='val_loss',
                                                                 min_delta=0.001,
                                                                 patience=2)],
                                      validation_data = val_generator,
                                      validation_steps = ceil(n_val_samples/batch_size))

        model_name = 'ssd300_0'
        model.save('./{}.h5'.format(model_name))
        model.save_weights('./{}_weights.h5'.format(model_name))
        print()
        print("Model saved as {}.h5".format(model_name))
        print("Weights also saved separately as {}_weights.h5".format(model_name))
        print()

        ### Make predictions

        # 1: Set the generator
Execute_command = model()
Execute_command.make_model()
