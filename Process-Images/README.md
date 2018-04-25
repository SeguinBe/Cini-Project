# Pipeline

- pipeline.py (cardboard + image extraction)

```
EXPORT OMP_NUM_THREADS=4
python pipeline.py -r /mnt/project_cini/jpg/ -m /scratch/benoit/tensorboard_docseg/vgg_test_3/export -d /mnt/cluster-nas/benoit/cini_processed -l /home/seguin/cini_image_extraction_final.log --skip-processed -g 1 -w 3
```

- pipeline_text.py (OCR + label association)

```
python pipeline_text.py -d /scratch/benoit/cini_processed/ -w 8 -s -l /home/seguin/cini_text.log
```

- exports images

## Export images :

```
python export_images.py
     -d /mnt/cluster-nas/benoit/cini_processed/
          -i ~/project_replica/datasets/cini
               -c ~/project_cini/cardboards
```

## DHCanvas Export (deactivated)

Data should be imported by Orlin.
Elements are accessible from predictable names: http://cini.dhlab.epfl.ch/page/view/{collection_uuid}/p{page_number}.
**WARNING** : some numbers are reused 1A_345 and 1A_345A for instance....
For instance 1A_345 -> page_number=345 collection_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, 'DHCANVAS_CINI_{}'.format('1A')))
