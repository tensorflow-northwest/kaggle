# -*- coding: utf-8 -*-
from __future__ import print_function

#### Home Directory ###
import os, sys
from pathlib import Path

home = str(Path.home())
os.chdir(home)
os.makedirs('nasnet', exist_ok=True)
os.chdir('nasnet')
os.makedirs('nasnet_large/weights', exist_ok=True)
os.makedirs('nasnet_large/model', exist_ok=True)
os.makedirs('nasnet_large/data', exist_ok=True)
os.makedirs('nasnet_large/files', exist_ok=True)

if not os.path.isfile('image.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/virtualdvid/MachineLearning/master/keras/image.py','image.py')
if not os.path.isfile('nasnet.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/virtualdvid/MachineLearning/master/keras/nasnet.py','nasnet.py')

### Libraries ###
 
import time
start_time = time.time()

import h5py, random, pickle, glob
import numpy as np

#from keras.preprocessing.image import ImageDataGenerator
from image import ImageDataGenerator
from keras.utils import np_utils, Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.optimizers import Adam
from nasnet import NASNet, NASNetLarge, preprocess_input
from keras import backend as K
from keras.models import load_model
import subprocess

K.set_image_dim_ordering('tf')

### Fuctions ###
def elapsed(start):
    """
    Returns elapsed time in hh:mm:ss format from start time in unix format
    """
    elapsed = time.time()-start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

print('Begin execution...')

'''# DevCloud Optimization
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': 6})
session = tf.Session(config=config)
K.set_session(session)

# Set OpenMP* environment variables (OMP_) and extensions (KMP_).
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"'''

### Model ###
weights_file = 'nasnet_large/weights/NASNet-iMaterialist_large.h5'
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.5), cooldown=0, patience=3, min_lr=0.5e-5)
csv_logger = CSVLogger('nasnet_large/NASNet-iMaterialist_large.csv')
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_predictions_acc', save_best_only=True,
                                   save_weights_only=True, mode='max')
TensorBoard = TensorBoard(log_dir='./nasnet_large/graph', histogram_freq=0, write_graph=True, write_images=True)

#load model
nb_classes = 128
#Build the model using the auxilary branch to correctly train NASNet
model = NASNetLarge((224, 224, 3), 
					dropout=0.5,
					use_auxiliary_branch=True,
					include_top=True,
					weights=None,
					classes=nb_classes)
print('Loaded master model')
#load weights
if not os.path.isfile(weights_file):
	model.load_weights('nasnet_large/weights/NASNet-large.h5', by_name=True, skip_mismatch=True)
	print('Loaded imagenet weights')
else:
	weight_files = glob.glob('nasnet_large/weights/*_large.h5')
	last_weight_file = max(weight_files, key=os.path.getctime)
	model.load_weights(last_weight_file)#, by_name=True, skip_mismatch=True)
	print('Loaded last trained weights: {}'.format(last_weight_file))

### Get last lr ###
try:
    hist_files = glob.glob('nasnet_large/p_files/*_large.p')
    last_hist_file = max(hist_files, key=os.path.getctime)
    with open(last_hist_file, 'rb') as fp:
        history = pickle.load(fp)
    #last_lr = history['lr'][-1]
    last_lr = np.mean([history['lr'][-1],1e-3])
    print('\nLast learning Rate = {}'.format(last_lr))
except:
    last_lr = 0.001
    print('\nLast learning Rate = {}'.format(last_lr))
	
### Compile ###
optimizer = Adam(lr=last_lr, clipnorm=5) #1e-3
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
			  optimizer=optimizer, metrics=['accuracy'], loss_weights=[1.0, 0.4])

### AUGMENTATION ###

# This will do preprocessing and realtime data augmentation:
# data generator for train set
train_datagen = ImageDataGenerator(#horizontal_flip=True,
                        #brightness_range=[1.2,1.2], # 0.5>val<=2 is not working
                        preprocessing_function=preprocess_input
                        #preprocessing_function=imgaug_steroids
                        )

# data generator for test set
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

### class_weights ###
label_wts = np.load('nasnet_data/label_wts.npy')
label_wts = np.ndarray.tolist(label_wts)

### loading file ###

# Link training data
scratch_dir = sys.argv[1]
print ('\nSCRATCH_DIR: {}\n'.format(scratch_dir))

train_files = os.path.join(scratch_dir,'files/train_tr')
val_files = os.path.join(scratch_dir,'files/train_val')

# generator for reading train data from folder
train_generator = train_datagen.flow_from_directory(
    train_files,
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = 200,
    class_mode = 'categorical')

# generator for reading validation data from folder
validation_generator = validation_datagen.flow_from_directory(
    val_files,
    target_size = (224, 224),
    color_mode = 'rgb',
    batch_size = 200,
    class_mode = 'categorical')

### Training ###
history = model.fit_generator(generator = train_generator, #(X, [y,y])
                            epochs = 8,
                            initial_epoch = 0,
                            validation_data = validation_generator, #(X, [y,y])
                            verbose = 1,
                            workers = 5,
                            use_multiprocessing = True,
                            steps_per_epoch = 100, #750
                            validation_steps = 25,
                            class_weight = [label_wts, label_wts],
                            #max_queue_size = 2,
                            callbacks=[lr_reducer, csv_logger, model_checkpoint, TensorBoard])

# Save weights, model and history
trained_weights_file = 'nasnet_large/weights/NASNet-trained-weights-{}_large.h5'.format(time.strftime("%Y%m%d-%H%M%S"))

model.save_weights(trained_weights_file)

print('\nSaved weights: ', trained_weights_file)

history_file = 'nasnet_large/p_files/history-{}_large.p'.format(time.strftime("%Y%m%d-%H%M%S"))
with open(history_file, 'wb') as fp:
    pickle.dump(history.history, fp, protocol=pickle.HIGHEST_PROTOCOL)
print('\nSaved history file: {}'.format(history_file))

print('\nElapsed time: {}'.format(elapsed(start_time)))

print('\nRestarting!')
os.chdir(home)
subprocess.call('../mj_start_l.sh', shell=True)
