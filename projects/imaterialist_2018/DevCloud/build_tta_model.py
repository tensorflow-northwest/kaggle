# -*- coding: utf-8 -*-
"""devcl-predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ceMVNARCs6KsS4VHn8RapmnD7UpIw5TX
"""

from __future__ import print_function
# from __future__ import absolute_import

import h5py, time, os, sys, argparse

from keras.models import load_model, Sequential, Model
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Flatten, InputLayer, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

def elapsed (start):
    elapsed = time.time()-start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def static_zoom(x, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='constant', cval=0.):
    zx, zy = zoom_range
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def static_shift(x, shift, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='constant', cval=0.):
    hs, ws = shift
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hs * h
    ty = ws * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
  
def static_shear(x, intensity, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='constant', cval=0.):
    shear = np.deg2rad(intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x  

def static_rotation(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.):
    theta = np.deg2rad(rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
  
def flip_horiz(x, flip_axis = 0):
    np.flip(x,flip_axis)
    return x

def dummy_func(x, dummy_arg):
    return x
  
def apply_transform(x,
                    transform_matrix,
                    channel_axis=2,
                    fill_mode='constant',
                    cval=0.):
    x = np.moveaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x
  
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
  
class ThreadedModel:
    def __init__(self, model_path, train_mean = None, train_std = None,
                 weights = None, bs = 100, nasnet = True):
        'Loads model, then predicts on dummy data so keras can build GPU function'

        self.model = load_model(model_path)
        self.graph = tf.get_default_graph
        if weights:
            self.model.load_weights(weights)
        self.train_mean = train_mean
        self.train_std = train_std
        self.nasnet = nasnet
        self.bs = bs
        self.simple_scale = train_mean is None or train_std is None
        self.model.predict(np.zeros(shape=(self.bs,224,224,3),
                                    dtype=np.float32), batch_size=self.bs)
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()

    def preproccesing(self, X):
        'Scale data conditionally'
        X = X.astype('float32', copy=False)
        if self.simple_scale:
            X /= 127.5
            X -= 1
        else:
            X -= self.train_mean
            X /= self.train_std
        return X

    def frozen_predict(self, data):
        'Runs predict method of model'
        X_scale = self.preproccesing(data)
        with self.session.as_default():
            with self.graph.as_default():
                if self.nasnet:
                    return self.model.predict(X_scale, batch_size = self.bs)[0]
                else:
                    return self.model.predict(X_scale, batch_size = self.bs)

def get_transform_preds(model_file,
                        weights = None,
                        data_file = 'imMaterialist/all_images.h5',
                        num_images = 6000,
                        batch_size = 100,
                        swap_classes = True,
                        train_mean = None, train_std = None,
                        img_offset = 0,
                        transforms = [dummy_func],
                        tr_vals = ['asdf'],
                        faster = False,
                        nasnet = True,
                        verbose = 0):
    '''
    Predicts images on a frozen graph of given model over given transformations.
    Can handle multi-threading if num_threads > 1.
    Scales by training mean and std if provided, otherwise by keras default.
    
    Returns ndarray of class probabilities with dimensions:
        (num_images, num_transformations, num_classes, 1)
    '''
        
    full_start = time.time()
    predictions = [None]*len(transforms)
    
    ####################### USED FOR FOR FASTER VERSION ######################
    if faster:
        K.clear_session()
        xmodel = ThreadedModel(model_file,train_mean,train_std, weights = weights)
    ##########################################################################

    with h5py.File(data_file, 'r') as hf:
        # Open connection to images
        images = hf['test']['images'][img_offset:num_images].astype(np.uint8, copy = False)
        
        if faster and verbose > 0:
            print('Loaded data and model in:', elapsed(full_start))
        data_start = time.time()

        
        # Predict on each batch of image transforms
        for j, transform in enumerate(transforms):

            pred_tr_start = time.time()
            
            ################### DONT USE FOR FASTER VERSION ################
            if not faster:
                K.clear_session()
                xmodel = ThreadedModel(model_file,train_mean,train_std,
                                       weights = weights)
                if verbose > 0:
                    if j == 0:
                        print('Loaded data and model in:', elapsed(full_start))
                    else:
                        print('Reloaded model in:', elapsed(pred_tr_start))
            ################################################################
            
            tr_images = np.array([transform(img,tr_vals[j]) for img in images])
            predictions[j] = xmodel.frozen_predict(data = tr_images)
            if not faster:
                del xmodel

            if verbose > 0:
                print('Predictions for transform number {} -- complete! Time: {}'\
                      .format(j+1, elapsed(pred_tr_start)))

    if verbose > 0:
        print('Total time:',elapsed(full_start))
    
    full_preds = [p.reshape(num_images - img_offset,1,128) for p in predictions]
    matrix_preds = np.concatenate(full_preds,1)
    if swap_classes:
        matrix_preds = matrix_preds[:, :, [127, *np.arange(1,127), 0]]
    matrix_preds = matrix_preds.reshape((*matrix_preds.shape,1))
    if faster:
        del xmodel
    return matrix_preds

def build_tta_model(pred_mat, dest, dest_path = None, data_file = None,
                   img_offset = 0, num_images = 6000, tta_form = None,
                   verbose = 0):
    
    assert pred_mat.shape[0] == num_images
    K.clear_session()
    dest = dest_path + dest if dest_path else dest
    
    # get test labels and convert to one-hot encoded
    with h5py.File(data_file, 'r') as hf:
        test_labels = hf['test']['labels'][img_offset:(img_offset + num_images)]
    test_labels = np.reshape(test_labels, (len(test_labels), 1)).astype(np.uint8)
    test_labels -= 1
    test_labels = np_utils.to_categorical(test_labels, 128)

    # initialize callbacks
    model_checkpoint = ModelCheckpoint(dest, monitor = 'val_acc',
                                       save_best_only=True,
                                       mode = 'max',verbose = verbose > 1)
    early_stop = EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 200)
    lr_reducer = ReduceLROnPlateau(monitor='val_acc', mode = 'max',
                                   factor = np.sqrt(0.5), patience = 10,
                                  min_lr = 1e-8)
    
    # build or load model
    if tta_form:
        model = load_model(tta_form)    
    else:    
        model = Sequential()
        model.add(InputLayer(input_shape=pred_mat.shape[1:]))
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(.8))
        model.add(Dense(128, activation = 'softmax'))

    model.compile(optimizer=Adam(lr = 1e-2),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(pred_mat, test_labels, batch_size = 500, epochs = 1000,
                         callbacks = [model_checkpoint, early_stop, lr_reducer],
                         validation_split = .18, verbose = verbose > 1)
    if verbose > 0:
        print('Model has been trained and saved to {}.\nBest Val_acc: {}'.format(dest, model_checkpoint.best))

start_time=time.time()

# initiates parser with program description for help argument.
text = "Generates prediction probability matrices for various tranformations on\
        labelled images. Then uses probability matrices, to train a model for \
        predicting classes, and saves the model to an h5 file. Assumes images \
        are stored in an h5 file structured to have 'images' and 'labels' as \
        the keys to group 'test' where images are stored."
parser = argparse.ArgumentParser(description = text)  

parser.add_argument('--dest', help = 'File name for the TTA model, ie tta.h5',
                   default = None)
parser.add_argument('--dest_path', help = 'Path to use for TTA model, defaults \
to nasnet/predictions/', default = 'nasnet/predictions/')
parser.add_argument('--save_mode', help = 'Whether to only save raw predictions, model or both',
                    choices = ['model', 'full', 'both'], default = 'model')
parser.add_argument('--model_file', help = 'Image classifier as keras model',
                   default = 'iMaterialist-model.h5')
parser.add_argument('--model_path', help = 'Path to classifieer',
                   default = 'nasnet/')
parser.add_argument('--model_weights', help = "Name of weights file for model \
to load if needed. If not specified, no weights will be loaded. Assumes \
file is saved in a directory named 'weights' within the model path",
                    default = None)
parser.add_argument('--raw_file', help = 'File path and name to save prediction \
probability matrices under (if desired). If [p][preprocessed] is true, this file \
should contain preprocomputed class probility vectors. If nothing specified, \
will saved as raw_test_<timestamp>.npy in nasnet/predictions/raw_predictions',
                    default = None)
parser.add_argument('--tta_form', help = 'Path and file name of an untrained \
keras model to train on test images. Defaults to simple seqential model with a \
single hidden layer and dropout.', default = None)
parser.add_argument('-p', '--preprocessed', help = 'If raw predictions already \
calculated, will expect to find them in raw_file.',
                    action = 'store_true', default = False)
parser.add_argument('--data_file', help = 'Path and file of images.',
                   default = "nasnet/data/all_images.h5")
parser.add_argument('--num_images', help = 'The number of images to run \
predictions on.', type = int, default = 6000)
parser.add_argument('--img_offset', help = 'Starting index for images \
predictions on (in case of memory limitations).', type = int, default = 0)
parser.add_argument('--batch_size', help = 'Batch size for model predict method',
                    type = int, default = 100)
parser.add_argument('-s','--swap_classes', help = 'Whether to swap prediction \
columns for class 1 and 128 (bug fix for some model versions)',
                   action = 'store_true', default = False)
parser.add_argument('--train_mean', help = 'File to upload for mean pixel values \
of training images. Assumes path nasnet/data/. Must also specify file for \
train_std. Defaults to keras preprocessing.', default = None)
parser.add_argument('--train_std', help = 'File to upload for st. dev of \
training image pixel values. Assumes path nasnet/data/. Must also specify file \
for train_mean. Defaults to keras preprocessing.', default = None)
parser.add_argument('-t','--transforms', help = 'Image transformations to be \
used, given as a space separated function names. Length must match \
tr_vals argument. Will also predict on original images regardless of transforms \
requested. Default (strongly encouraged) is a set of 14 transforms. This also \
requires that tta_model was built around these transforms.',
                    nargs = '+', default = None)
parser.add_argument('--tr_vals', help = "Arguments (in order) for functions \
given in transforms, as space separated values. Be careful, some functions take \
tuple arguments which should be given as strings, ie '(a,b)'. Length must match \
transforms. Strongly discourage using this before close examination of code.",
                    nargs = '+', default = None)
parser.add_argument('-v', '--verbose', help = 'Verbosity of program. 0 is silent,\
 1 will report on all transforms and major steps. 2 extends 1 to include \
 reporting from individual threads.', type = int, default = 0)
parser.add_argument('-f','--faster', help = 'Speed up predictions \
at the risk of stability)', action = 'store_true', default = False)

# read arguments from the command line and assign appropriate values
args = parser.parse_args()

if args.transforms is None:
    transforms = [dummy_func, flip_horiz, *[static_shift]*4,
                  *[static_rotation]*4, *[static_shear]*2, *[static_zoom]*3]
else:
    transforms = [dummy_func] + [eval(t) for t in args.transforms]

if args.tr_vals is None:
    tr_vals = ['banana',0, (.2,0.0), (0.0,.2),(-.2,0.0), (0.0,-.2), 20,-20,
               40, -40, 15, -15, (.8,.8), (.8,1.0), (1.0,.8)]
else:
    tr_vals = ['asdf'] + [eval(v) for v in args.tr_vals]

data_path = 'nasnet/data/'
train_mean = np.load(data_path + args.train_mean) if args.train_mean else None
train_std = np.load(data_path + args.train_std) if args.train_std else None

weights = '{}weights/{}'.format(args.model_path, args.model_weights) if args.model_weights else None
model_file = args.model_path + args.model_file
    
assert len(transforms) == len(tr_vals)
assert args.batch_size <= args.num_images
assert tf.device('/CPU:0')

if args.preprocessed:
    args.save_mode ='model'
    raw_preds = np.load(raw_file)
else:
    raw_preds = get_transform_preds(model_file=model_file, weights=weights,
                        data_file=args.data_file,
                        num_images=args.num_images,
                        batch_size=args.batch_size,
                        swap_classes=args.swap_classes,
                        train_mean=train_mean, train_std=train_std,
                        img_offset=args.img_offset, transforms=transforms,
                        tr_vals=tr_vals, verbose=args.verbose)

if args.save_mode != 'model':
    raw_path = 'nasnet/predictions/raw_predictions'
    os.makedirs(raw_path, exist_ok=True)
    raw_file = args.raw_file if args.raw_file else raw_path + '/raw_test_' + time.strftime('%m%d_%H%M')
    np.save('{}.npy'.format(raw_file),raw_preds)
    print('Raw predictions saved to', raw_file)

if args.save_mode != 'raw':
    build_tta_model(raw_preds, dest = args.dest, dest_path = args.dest_path,
                   data_file = args.data_file, img_offset = args.img_offset,
                   num_images = args.num_images, tta_form = args.tta_form,
                   verbose = args.verbose)


if args.verbose > 0:
    print('Elapsed time: {}'.format(elapsed(start_time)))