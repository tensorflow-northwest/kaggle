# coding: utf-8
#from __future__ import print_function
#from __future__ import absolute_import


# # iMatreialist - Predictor

# ## TFNW

# # Predict and Create Submission file

#### Home Directory ###
import os, sys
from pathlib import Path

home = str(Path.home())
os.chdir(home)
os.makedirs('nasnet', exist_ok=True)
os.chdir('nasnet')
os.makedirs('weights', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('files', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

from keras.models import load_model
from nasnet.nasnet import preprocess_input

import tensorflow as tf

import numpy as np
import pandas as pd
import h5py, time, os, sys

size = 224

start_time=time.time()

def elapsed (start):
    """
    Returns elapsed time in hh:mm:ss format from start time in unix format
    """
    elapsed = time.time()-start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

trained_weights = 'weights/NASNet-iMaterialist_v2.h5'
model_path = 'iMaterialist-model.h5'
missing = 'data/missing.csv'

if tf.device('/CPU:0'):
    print('Loading model takes about 3 minutes...')
    model=load_model(model_path)
    print('Model loaded. Loading weights...')
    model.load_weights(trained_weights)
    print('Weights file loaded: '+ trained_weights)

pred_df2 = pd.read_csv(missing, header=0)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

evaluate = False

if evaluate:
	from image import ImageDataGenerator
	# generator for reading validation data from folder
	validation_generator = test_datagen.flow_from_directory(
		'files/train_val',
		target_size = (224, 224),
		color_mode = 'rgb',
		batch_size = 200,
		class_mode = 'categorical')

	scores = model.evaluate_generator(validation_generator, steps=199, verbose=1)
	for score, metric_name in zip(scores, model.metrics_names):
		print("%s : %0.4f" % (metric_name, score))
		
else:
	from keras.preprocessing.image import ImageDataGenerator
	# generator for reading test data from folder
	test_generator = test_datagen.flow_from_directory(
		'files/test',
		target_size = (224, 224),
		color_mode = 'rgb',
		batch_size = 200,
		class_mode = 'categorical',
		shuffle = False)

	# test predictions with generator
	test_files_names = test_generator.filenames
	predictions = model.predict_generator(test_generator, steps=64, verbose=1)
	predictions = predictions[0].argmax(-1)
	predictions += 1
	pred_df = pd.DataFrame(predictions, columns = ['predicted'])
	pred_df.insert(0, 'id', test_files_names)
	pred_df['id'] = pred_df['id'].map(lambda x: x.lstrip('files/test\\').rstrip('.jpg'))
	pred_df['id'] = pd.to_numeric(pred_df['id'], errors = 'coerce')
	pred_df = pred_df.append(pred_df2, ignore_index=True)
	pred_df.sort_values('id', inplace = True)
	pred_df.fillna(83, inplace=True)
	filename = 'predictions/tfnw-preds_' + time.strftime('%Y%m%d_%H%M%S_%Z.csv')
	pred_df.to_csv(filename, index = False)
    
print('File {} saved.'.format(filename))

print('Elapsed time: {}'.format(elapsed(start_time)))
