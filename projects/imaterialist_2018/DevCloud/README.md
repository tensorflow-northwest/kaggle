## General Instructions

1. Upload model, datasets, trained weights and eval_dataset.h5 files
2. Adjust model_path, trained_weights and test_file variables accordingly.
3. Add your email address in im and devcloud-predict files
4. Run in DevCloud:
	```
	$ at 2:00am
	> qsub im
	> qsub devcloud-predict
	> <ctrl-d>
	```
---
## Script Documentation:

### build_tta_model.py

**_Description_**:
Generates prediction probability matrices for various tranformations on
labelled images. Then uses probability matrices, to train a model for
predicting classes, and saves the model to an h5 file. Assumes images are
stored in an h5 file structured to have 'images' and 'labels' as the keys to
group 'test' where images are stored.


|argument | alias | choices | default | description |
|---|---|---|---|---|
| --help |-h|||Show this help message and exit |
|--dest| | || File name for the TTA model, ie tta.h5 |
|--dest_path| ||nasnet/predictions/|Path to use for TTA model|
|--save_mode || model,full,both | model |Whether to only save raw predictions, model or both |
|--model_file | | | iMaterialist-model.h5 | Image classifier as keras model |
|--model_path | | | nasnet/ |Path to classifieer|
|--model_weights | | | | Name of weights file for model to load if needed. If not specified, no weights will be loaded. Assumes file is saved in a directory named 'weights' within the model path. |
|--raw_file | | | nasnet/predictions/raw_predictions/ | File path and name to save prediction probability |
| | | | raw_test_<timestamp>.npy | matrices under (if desired). If [p][preprocessed] is true, this file should contain preprocomputed class probility vectors. |
|--tta_form | | | |  Path and file name of an untrained keras model to train on test images. Defaults to simple seqential model with a single hidden layer and dropout.|
|--preprocessed | -p | | | If raw predictions already calculated, will expect to find them in raw_file.|
|--data_file | | | nasnet/data/all_images.h5 | Path and file of test images. Assumes h5 group ['test'] containing two datasets ['images'] and ['labels']. |
| --num_images| | | 6000 | The number of images to run predictions on. |
| --img_offset | | | 0 | Starting index of images to run predictions on (in case of memory limitations).|
|--batch_size | | | 100 | Batch size for model predict method |
|--swap_classes |-s| | | Whether to swap prediction columns for class 1 and 128 (bug fix for some model versions). |
|--train_mean | | | keras' preprocess | File to upload for mean pixel values of training images. Assumes path nasnet/data/. Must also specify file for train_std. |
|--train_std | | | keras' preprocess | File to upload for st. dev of training image pixel values. Assumes path nasnet/data/. Must also specify file for train_mean.|
|--transforms|-t| see note\* | see note\* | Image transformations to be used, given as a space separated function names. Length must match tr_vals argument. Will also predict on original images regardless of transforms requested. Default (strongly encouraged) is a set of 14 transforms. This also requires that tta_model was built around these transforms.|
|--tr_vals | | see note\* | see note\* | Arguments (in order) for functions given in transforms, as space separated values. Be careful, some functions take tuple arguments which should be given as strings, ie '(a,b)'. Length must match transforms. Strongly discourage using this before close examination of code.|
|--verbose| -v | | 0 | Verbosity of program. 0 is silent, 1 will report on all transforms and major steps. 2 extends 1 to include wall-of-text training report of 1k epochs.|
|--faster|-f| | | Speed up predictions at the risk of stability. *You've got to ask yourself one question: Do I feel lucky*|
	
**\* -- Note on --transforms and --tr_vals:**
These will default to a predetermined set of 15 static transforms (the first of which returns the original images unchanged):

>> `transforms = [dummy_func, flip_horiz, *[static_shift]*4,
>>                  *[static_rotation]*4, *[static_shear]*2, *[static_zoom]*3]`

>> `tr_vals = ['dummy_val',0, (.2,0.0), (0.0,.2),(-.2,0.0), (0.0,-.2), 20,-20,
>>           40, -40, 15, -15, (.8,.8), (.8,1.0), (1.0,.8)]`
	       
Any number or combination of transforms may be given, but order between transforms and tr_vals must match. Some transforms require only 1 argument, others require a tuple.

Example use `$python build_tta_model.py [other args] -t flip_horiz static_shift static_rotation --tr_vals 0 (.3, -.1) 20`
