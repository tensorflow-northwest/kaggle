1. Upload model, datasets, trained weights and eval_dataset.h5 files
2. Adjust model_path, trained_weights and test_file variables accordingly.
3. Add your email address in im and devcloud-predict files
4. Run in DevCloud:
	$ at 2:00am
	> qsub im
	> qsub devcloud-predict
	> <ctrl-d>
