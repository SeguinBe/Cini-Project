## Converting .cr2 to .jpg

With an anaconda installation :

- `conda env create -f environment.yml`
- `source activate process_raw`
- Then `python process_raw.py -r <raw_files_folder> -d <output_directory> -w <nb_workers> --md5`

A `.log` file is created at the root of the output directory.
A simple `cat <log-file> | grep ERROR` allows to see the files that were not properly processed (corrupted, wrong md5, etc...).

The `--md5` flag to perform md5 checking is not mandatory.

The `--verso` flag is to perform the conversion for verso files as well.


## problems left

- some md5 which are wrong
- some files without basename ('_recto.cr2')
- 46A being inside 44A and by himself  (corrected)