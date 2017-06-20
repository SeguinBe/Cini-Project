# Cini-Project

Project Structure
-----------------

Process-Raw:
*	`md5.py`: Contains functions to verify the md5 has
*	`process_raw.py`: Pipline that proceses Raw files and saves them as jpg

Process-Images:

*	`RawScan.py`: Class definition that contains methods for post processing
*	`pipeline.py`: Pipline that proceses the images
*	`unwarp.py`: Conatins functions and model to correct image warping

	Model:
	*	`model.py`: Tensorflow model definition
	*	`train_eval.py`: Conatins model train and evaluation functions
	*	`train_script.py`: Short script to train the model
	*	`upsample.py`: Definition of an unsample layer in Tensorflow
