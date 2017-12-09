# Metadata alignment

- First the metadata can be gathered as a big csv file with `gather_metadata.py -d <directory>`.

- Process them with openrefine and the `openrefine_commands.json` commands.

- Run `extract_artist_data.py` to extract the metadata from ULAN.

- Run `match_1st_pass.py -d <transformed-csv-file>` to perform the exact matching.

- Run `match_2nd_pass.py` to perform the approximate matching (will take some time).

- Run `match_final_pass.py -d <transformed-csv-file>` to generate the final correspondences dictionnary.

- Run `export_to_iiif_manifests.py <args>` that generates the IIIF manifests with corrected authors.


## IIIF Export

Data should go to `/srv/http/epfl.ch/dhlab/data/cini and cini_raw` on seguin@cdh-dhlabpc4.epfl.ch.
Then it is accessible from http://data.dhlab.epfl.ch/cini
