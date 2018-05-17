# -*- coding: utf-8 -*-
# gmlander

from __future__ import print_function
# from __future__ import absolute_import

import h5py, time, os, sys, argparse

from keras.models import load_model
from keras import backend as K

import tensorflow as tf

import threading as t
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

def elapsed (start):
    """
    Returns elapsed time in hh:mm:ss format from start time in unix format
    """
    elapsed = time.time()-start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def static_zoom(x, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='constant', cval=0.):
    """Performs a spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
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
    """Performs a spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        shift: Height and width shift as a tuple of float fractions
        of the height and width.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
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
    """Performs a spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
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
    """Performs a rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
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
    """Applies the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
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
                 weights = None, bs = 100, p_threads = 5, verbose = False):
        'Loads model, then predicts on dummy data so keras can build GPU function'

        self.model = load_model(model_path)
        if weights:
            self.model.load_weights(weights)
        self.train_mean = train_mean
        self.train_std = train_std
        self.bs = bs
        self.p_threads = p_threads
        self.verbose = verbose
        self.simple_scale = train_mean is None or train_std is None
        self.predictions = [None]*self.p_threads
        self.model.predict(np.zeros(shape=(self.bs,224,224,3),dtype=np.float32),
                           batch_size=self.bs)
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

    def thread_predict(self, data, thread_num):
        'Runs predict method of model'
        start_time = time.time()
        X_scale = self.preproccesing(data)
        if self.verbose:
            print('Thread {} took {} to preprocess'\
                  .format(thread_num,elapsed(start_time)))
        with self.session.as_default():
            with self.graph.as_default():
                self.predictions[thread_num] = \
                self.model.predict(X_scale, batch_size = self.bs)
                if self.verbose:
                    print('Thread {} complete. Total time: {}'\
                          .format(thread_num, elapsed(start_time)))
                    
                    
def get_transform_preds(model_file = 'xception-cut6-5.h5',
                        weights = None,
                        data_file = '../iMaterialist/eval_dataset.h5',
                        data_key = 'test_dataset',
                        num_images = 12801,
                        num_threads = 1,
                        batch_size = 100,
                        swap_classes = True,
                        train_mean = None, train_std = None,
                        img_offset = 0,
                        transforms = [dummy_func],
                        tr_vals = ['asdf'],
                        verbose = 0):
    '''
    Predicts images on a frozen graph of given model over given transformations.
    Can handle multi-threading if num_threads > 1.
    Scales by training mean and std if provided, otherwise by keras default.
    
    Returns ndarray of class probabilities with dimensions:
        (num_images, num_transformations, num_classes, 1)
    '''
    K.clear_session()
    full_start = time.time()
    predictions = [None]*len(transforms)

    xmodel = ThreadedModel(model_file,train_mean,train_std, weights = weights,
                           p_threads = num_threads, verbose = verbose > 1)

    with h5py.File(data_file, 'r') as hf:
        # Open connection to images
        if type(data_key) == str:
            images = hf['test_dataset']
        elif type(data_key) == list and len(data_key) == 2:
            images = hf[data_key[0]][data_key[1]]
        else:
            print('Uninterprettable input for data_key: {}.\nMust be a\
            string or list of 2 strings.'.format(data_key))
            del xmodel
            return None
        if verbose > 0:
            print('Loaded data and model in:', elapsed(full_start))
        data_start = time.time()

        # Create batch size based ranges for each thread.
        # Give last thread remainder of images.
        full_batches = num_images // batch_size
        batches_per_thread = full_batches // num_threads
        im_per_t = batches_per_thread * batch_size
        extra = num_images % (im_per_t)

        # start threads, each thread runs model predict on range of images
        for j, transform in enumerate(transforms):

            threads = [None] * num_threads
            pred_tr_start = time.time()

            for i in range(num_threads):
                start_ix = im_per_t*i + img_offset
                end_ix = im_per_t*(i + 1) + img_offset \
                    +max(0,i + 2 - num_threads)*extra # add extra to last thread
                threads[i] = t.Thread(target=xmodel.thread_predict,
                  kwargs={'data': np.array([transform(img,tr_vals[j]) for img \
                    in images[start_ix:end_ix].astype(np.uint8, copy = False)]),
                                            'thread_num': i})
                threads[i].start()

            # wait for threads to finish  
            for i in range(num_threads):
                threads[i].join()

            if verbose > 0:
                print('Predictions for transform number {} -- complete! Time: {}'\
                      .format(j+1, elapsed(pred_tr_start)))
            y_probs = np.concatenate(xmodel.predictions)
            predictions[j] = y_probs

    if verbose > 0:
        print('Total time:',elapsed(full_start))
    predictions = np.concatenate([np.reshape(p,(p.shape[0],1,128)) for p in predictions],1)
    if swap_classes:
        predictions = predictions[:, :, [127, *np.arange(1,127), 0]]
    predictions = predictions.reshape((*predictions.shape,1))
    del xmodel
    return predictions
start_time=time.time()

# initiates parser with program description for help argument.
text = 'Generates prediction probability matrices for various tranformations on\
        provided images. Then uses provided TTA model to predict classes from\
        probability matrices, and saves the predictions to a csv file.'
parser = argparse.ArgumentParser(description = text)  

parser.add_argument('--model_file', help = 'Image classifier as keras model',
                   default = 'iMaterialist-model.h5')
parser.add_argument('--model_path', help = 'Path to classifieer',
                   default = 'nasnet/')
parser.add_argument('--model_weights', help = "Name of weights file for model \
to load if needed. If not specified, no weights will be loaded. Assumes \
file is saved in a directory named 'weights' within the model path",
                    default = None)
parser.add_argument('--raw_file', help = 'File name to save prediction \
probability matrices under (if desired).', default = None)
parser.add_argument('--tta_model', help = 'Name of model to generate predictions \
from probability matrices. Default assumes it is stored in predictions directory \
of model_path.', default = 'tta_model.h5')
parser.add_argument('--data_file', help = 'Path and file of images.',
                   default = "nasnet/data/eval_dataset.h5")
parser.add_argument('--data_key', help = 'h5 key (or list of 2 keys) to access \
images in the data_file', nargs = '+', default = 'test_dataset')
parser.add_argument('--num_images', help = 'The number of images to run \
predictions on.', type = int, default = 12801)
parser.add_argument('--img_offset', help = 'Starting index for images \
predictions on (in case of memory limitations).', type = int, default = 0)
parser.add_argument('--num_threads', help = 'Number of threads to use for \
making predictions. At the cost of memory, can speed up operations that involve \
extensive computations outside of the interpretter.',
                    type = int, default = 1)
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
parser.add_argument('--missing_fill', help = 'Class label to predict for \
missing images. Default will be a sequence of random values.',
                    type = int, default = None)

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

weights = '{}weights/{}'(args.model_path, args.model_weights) if args.model_weights else None
model_file = args.model_path + args.model_file
    
assert len(transforms) == len(tr_vals)
assert args.batch_size <= args.num_images

if tf.device('/CPU:0'):
    raw_preds = get_transform_preds(model_file=model_file, weights=weights,
                        data_file=args.data_file, data_key=args.data_key,
                        num_images=args.num_images,
                        num_threads=args.num_threads,
                        batch_size=args.batch_size,
                        swap_classes=args.swap_classes,
                        train_mean=train_mean, train_std=train_std,
                        img_offset=args.img_offset, transforms=transforms,
                        tr_vals=tr_vals, verbose=args.verbose)

if args.raw_file:
    raw_path = 'nasnet/predictions/raw_predictions'
    os.makedirs(raw_path, exist_ok=True)
    np.save('{}/{}.npy'.format(raw_path,args.raw_file),raw_preds)
    
model = load_model('nasnet/predictions/{}'.format(args.tta_model))
tta_pred = model.predict(raw_preds, batch_size = 500, verbose = 1)
tta_pred = tta_pred.argmax(1)
tta_pred += 1

df = pd.DataFrame({'id': np.arange(args.img_offset,
                                   args.img_offset + len(tta_pred)), 
                   'predicted' : tta_pred})

with h5py.File(args.data_file, 'r') as hf:
    missing = hf['missing'][:]

# could have skipped this filter step and sliced hf on those indices, but
# wanted to protect against missing id's being out of order
missing = [m for m in missing if args.img_offset <= m <= (args.img_offset + args.num_images)]
if args.missing_fill:
    df.iloc[missing, 1] = args.missing_fill
else:
    df.iloc[missing, 1] = np.random.randint(1,129, size = len(missing))

df_missing.drop(index=0,inplace = True)


os.makedirs('nasnet/predictions', exist_ok=True)
filename = 'nasnet/predictions/tfnw-preds_' + time.strftime('%Y%m%d_%H%M%S_%Z.csv')
df.to_csv(filename, index=False)

if args.verbose > 0:
    print('Sample predictions:\n',df.head(10))
    print('File {} saved.'.format(filename))
    print('Elapsed time: {}'.format(elapsed(start_time)))
