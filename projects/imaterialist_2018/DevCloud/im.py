# -*- coding: utf-8 -*-
from __future__ import print_function

"""
# NASNet-iMaterialist
## TFNW

Train NASNet on the iMaterialist dataset.

- Sequence class

+ Augmentation


@alkari

@virtualdvid
"""


"""## Setup """

import os
import urllib.request
from pathlib import Path

home = str(Path.home())
os.chdir(home)
os.makedirs('nasnet', exist_ok=True)
os.chdir('nasnet')
os.makedirs('weights', exist_ok=True)
os.makedirs('data', exist_ok=True)

"""### Download model definition"""

if not os.path.isfile('nasnet.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alkari/Keras-NASNet/master/nasnet.py',
                               'nasnet.py')

if not os.path.isfile('weights/NASNet-large.h5'):
    urllib.request.urlretrieve('https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large.h5',
                               'weights/NASNet-large.h5')

"""### Imports"""

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from nasnet.nasnet import NASNet, NASNetLarge, preprocess_input
from random import randrange
import numpy as np
import h5py
import random
import time
import sys


# DevCloud Optimization
import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=24, inter_op_parallelism_threads=2, allow_soft_placement=True,  device_count = {'CPU': 24})
session = tf.Session(config=config)

# Set OpenMP* environment variables (OMP_) and extensions (KMP_).
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

# Set session
tf.keras.backend.set_session(session)

# Helper functions
def elapsed (start):
    """
    Returns elapsed time in hh:mm:ss format from start time in unix format
    """
    elapsed = time.time()-start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def cutout(img):
    MAX_CUTS = 5  # chance to get more cuts
    MAX_LENGTH_MULTIPLIER = 8  # change to get larger cuts ; 16 for cifar 10, 8 for cifar 100
    
    height = 224
    width = 224

    mask = np.ones((height, width,3), np.float32)
    nb_cuts = np.random.randint(0, MAX_CUTS + 1)

    for i in range(nb_cuts):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, MAX_LENGTH_MULTIPLIER + 1)

        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)

        mask[y1:y2, x1:x2] = 0.

    # apply mask
    img = img * mask

    return img

"""### Parameters"""

weights_file = 'weights/NASNet-iMaterialist.h5'
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.5), cooldown=0, patience=5, min_lr=0.5e-5)
csv_logger = CSVLogger('NASNet-iMaterialist.csv')
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_predictions_acc', save_best_only=True,
                                   save_weights_only=True, mode='max')
batch_log = 'batch_log.csv'

nb_classes = 128

# input image dimensions
img_rows, img_cols = 224, 224
img_channels = 3

"""## Build the model using the auxilary branch to correctly train NASNet"""

model = NASNetLarge((img_rows, img_cols, img_channels),dropout=0.5, use_auxiliary_branch=True, include_top=True, weights=None, classes=nb_classes)

# first time only, load imagenet pre-trained weights
if not os.path.isfile(weights_file):
    if not os.path.isfile('weights/NASNet-large.h5'):
        urllib.request.urlretrieve('https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large.h5',
                                   'weights/NASNet-large.h5')
    model.load_weights('weights/NASNet-large.h5', by_name=True, skip_mismatch=True)
    print('Loaded imagenet weights')
else:
    model.load_weights(weights_file, by_name=True, skip_mismatch=True)
    print('Loaded last trained weights')

optimizer = Adam(lr=1e-3, clipnorm=5)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
              optimizer=optimizer, metrics=['accuracy'], loss_weights=[1.0, 0.4])

"""## Feed Generator"""

def batch_generator(path, batch_size, data_set='train', seed=101, data_augmentation=False):
    assert data_set in ['train','test'] # Post test phase validation

    batch_start = 0
    batch_end = batch_size
    log = []
    log.append(['file','start','end', 'time'])

    # File iterator
    filelist = []
    for file in os.listdir(path)[:]:
        if file.startswith(data_set+"_"):
            filelist.append(os.path.join(path,file))
    random.seed(seed)
    random.shuffle(filelist)  

    file_count = len(filelist)
    files = iter(filelist)

    # Generate batches
    while True:
        if batch_start == 0:
            with open(batch_log,'a+') as b_log:
                for i in range(len(log)):
                    b_file = log[i][0]
                    b_start = log[i][1]
                    b_end = log[i][2]
                    e_time = log[i][3]
                    b_log.write("{},{},{},{}\n".format(b_file, b_start, b_end, e_time))
            log = []

            file = next(files) # Select next file
            hf = h5py.File(file, 'r')
            file_len = len(hf[data_set]['images'][:])
            assert file_len == 1000

        limit = min(batch_end, file_len)

        # Load next batch of images  
        X = hf[data_set]['images'][batch_start:limit]
        if data_augmentation:
            X = augmentation(X, batch_size)
            for i, X in enumerate(X):
                if i == 1:
                    break
        else:
            X = np.asarray(X)
            X = X.astype('float32', copy=False)
            X = preprocess_input(X)

        label = hf[data_set]['labels'][batch_start:limit]
        Y = np.asarray(label, dtype='int8')

        Y = np.asarray(Y)
        Y = np.reshape(Y, (len(Y), 1))
        Y = np_utils.to_categorical(Y, nb_classes)

        yield (X,[Y,Y])

        # Write batch_log for each batch
        log.append([str(file.split("/")[5]), str(batch_start), str(limit), elapsed(start_time)])

        batch_start += batch_size   
        batch_end += batch_size  

        if batch_start % file_len == 0: # files contain 1000 images each
            hf.close()
            batch_start = 0
            batch_end = batch_size
            file_count -= 1
            # All files loaded. Shuffle files and load a new filelist
            if file_count < 1:
                file_count = len(filelist)
                seed +=random.randint(1,10)
                random.seed(seed)
                random.shuffle(filelist)        
                files = iter(filelist)

"""## Image Augmentation"""

def augmentation(X, batch_size = 20, shuffle=False):
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
      #featurewise_center=False, # set input mean to 0 over the dataset
      #samplewise_center=False, # set each sample mean to 0
      #featurewise_std_normalization=False, # divide inputs by std of the dataset
      #samplewise_std_normalization=False, # divide each input by its std
      #zca_whitening=True, # is not working
      #zca_epsilon=1e-06, # is not working
      rotation_range=5.0, # randomly rotate images in the range (degrees, 0 to 180)
      #width_shift_range=0.02, # randomly shift images horizontally (fraction of total width)
      height_shift_range=0.02, # randomly shift images vertically (fraction of total height)
      brightness_range=[1.2,1.2], # 0.5>val<=2 pending for test
      #shear_range=1.0,
      #zoom_range=1.0,
      #channel_shift_range=0.0,
      #fill_mode='nearest',
      #cval=0.0,
      horizontal_flip=True, # randomly flip images
      #vertical_flip=False, # randomly flip images
      #rescale=1./127.5, #1./255 # =preprocess_input
      #data_format=None,
      #validation_split=0.0,
      preprocessing_function=preprocess_input, ########Comment out this line while your are doing the test!
      #preprocessing_function=cutout # randomly apply cutout
      )  

  # Compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X)

    X_aug = datagen.flow(X, batch_size=batch_size, shuffle=shuffle)

    return X_aug

"""## Begin Training"""

#if __name__ == "__main__":
    
start_time = time.time()

# Check input
if len(sys.argv) < 2:
    data_augmentation = False
elif sys.argv[1]=="A":
    data_augmentation = True
else:
    print ('Usage: python', sys.argv[0], '"A"')
    sys.exit(1)
seed = random.randint(1,100000)

path = os.path.join(home,"nasnet/data")
nb_train_files = 0
nb_test_files = 0
for file in os.listdir(path)[:]:
    if file.startswith("train_"):
        nb_train_files += 1
    elif file.startswith("test_"):
        nb_test_files += 1

images_per_file = 1000
batch_size = 20
nb_epoch = 1 #  (1 = 10 epochs)
files_per_group = 10
nb_steps = (images_per_file / batch_size) * files_per_group

if data_augmentation:
    print('Using real-time data augmentation.')
else:
    print('Not using data augmentation.')

history = model.fit_generator(batch_generator(path, batch_size, 'train', seed=seed,
                                              data_augmentation=data_augmentation),
          epochs=14, # nb_train_files * nb_epoch,
          #initial_epoch = 10,
          validation_data=(batch_generator(path, batch_size, 'test')),
          #validation_data=(X_test, [Y_test, Y_test]),
          shuffle=True,
          verbose=2,
          #workers=2, 
          use_multiprocessing=True,
          steps_per_epoch = nb_steps,
          validation_steps = (images_per_file / batch_size ) * nb_test_files,
          #max_queue_size=1,
          callbacks=[lr_reducer, csv_logger, model_checkpoint])

#scores = model.evaluate(X_test, [Y_test, Y_test], batch_size=batch_size)
#for score, metric_name in zip(scores, model.metrics_names):
#    print("%s : %0.4f" % (metric_name, score))
print('\n')

print('Elapsed time: {}'.format(elapsed(start_time)))

model.save('iMaterialist-trained-model.h5', include_optimizer=True)
