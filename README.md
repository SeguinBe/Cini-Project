# Cini-Project

Project Structure
-----------------

Process-Raw:

* `md5.py`: Contains functions to verify the md5 has
* `process_raw.py`: Pipline that proceses Raw files and saves them as jpg

Process-Images:

* to be updated


`python pipeline_text.py -d /scratch/benoit/cini_processed/ -w 8 -s -l /home/seguin/cini_text.log`

`python pipeline.py -r /scratch/benoit/cini_full_images/ -m /scratch/benoit/tensorboard_docseg/5_layers/export -d /scratch/benoit/cini_processed/ -w 6 -l /home/seguin/cini_image_extraction.log`