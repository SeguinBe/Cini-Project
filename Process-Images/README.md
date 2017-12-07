# Pipeline

- pipeline.py (cardboard + image extraction)

```
python pipeline.py -r /mnt/cluster-nas/benoit/cini_full_images/ -m /scratch/benoit/tensorboard_docseg/vgg_test_3/export -d /scratch/benoit/cini_processed_3/ -l /home/seguin/cini_image_extraction_3.log --skip-processed -g 0 -w 3
```

- pipeline_text.py (OCR + label association)

```
python pipeline_text.py -d /scratch/benoit/cini_processed/ -w 8 -s -l /home/seguin/cini_text.log
```

- exports (images, dhcanvas, iiif manifests)



## IIIF Export

Data should go to `/srv/http/epfl.ch/dhlab/data/cini` on seguin@cdh-dhlabpc4.epfl.ch.
Then it is accessible from http://data.dhlab.epfl.ch/cini

## DHCanvas Export (deactivated)

Data should be imported by Orlin.
Elements are accessible from predictable names: http://cini.dhlab.epfl.ch/page/view/{collection_uuid}/p{page_number}.
**WARNING** : some numbers are reused 1A_345 and 1A_345A for instance....
For instance 1A_345 -> page_number=345 collection_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, 'DHCANVAS_CINI_{}'.format('1A')))


## Export images :

```
python export_images.py
    -d /scratch/benoit/cini_processed/
    -i /home/seguin/NAS-home/datasets/cini/
    -c /mnt/cluster-nas/benoit/cini-cardboards
```
