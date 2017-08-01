# Pipeline

- pipeline.py (cardboard + image extraction)
- pipeline_text.py (OCR + label association)
- exports (images, dhcanvas, iiif manifests)



## IIIF Export :

Data should go to `/srv/http/epfl.ch/dhlab/data/cini` on seguin@cdh-dhlabpc4.epfl.ch.
Then it is accessible from http://data.dhlab.epfl.ch/cini

## DHCanvas Export :

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
