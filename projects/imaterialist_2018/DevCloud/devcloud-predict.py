# coding: utf-8
#from __future__ import print_function
#from __future__ import absolute_import


# # iMatreialist - Predictor

# ## TFNW

# @alkari

# # Predict and Create Submission file


from keras.models import load_model

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

trained_weights = 'nasnet/weights/NASNet-iMaterialist.h5'

model_path = 'nasnet/iMaterialist-model.h5'

test_file = "nasnet/data/eval_dataset.h5"


if tf.device('/CPU:0'):
  model=load_model(model_path)
  print('Model loaded. Loading weights...')
  model.load_weights(trained_weights)
  print('Weights file loaded: '+ trained_weights)

def load_final_testset(filename,max=None):
    with h5py.File(filename, 'r') as hf:
        test_images = hf['test_dataset'][:max]        
        # preprocess
        test_images = test_images.astype('float32', copy=False)
        test_images /= 127.5
        test_images -=1 

        missing = hf['missing'][:]
    return test_images,missing

# Load test set
x_test_final,missing = load_final_testset(test_file)

# Predict in 100 batches
predictions = model.predict(x_test_final, verbose=1, batch_size=100)
predictions = predictions[0].argmax(-1)

# Adjust for files force labeled 0 to 128 and label missing files.
arbitrary = 83 # or random!
for i in range(predictions.shape[0]):
    if predictions[i]==0:
        predictions[i]=128
        print('Changed image {} from 0 to {}'.format(i,predictions[i]))
    if i in missing:
        predictions[i]=arbitrary
        print('Changed missing image {} prediction to {}'.format(i,predictions[i]))

def save_preds(predictions):
    
    df = pd.DataFrame({'id' : range(len(predictions)), 
                       'predicted' : predictions})
    df = df.drop(0) # Drop none-existent image[0] prediction
    os.makedirs('nasnet/predictions', exist_ok=True)
    filename = 'nasnet/predictions/tfnw-preds_' + time.strftime('%Y%m%d_%H%M%S_%Z.csv')
    df.to_csv(filename, index=False)
    return(filename)

filename = save_preds(predictions)


for i in range(20):
    print(i, predictions[i])
    
print('File {} saved.'.format(filename))

print('Elapsed time: {}'.format(elapsed(start_time)))
