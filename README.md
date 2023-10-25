# Azevedo_Lesser_Phelps_Mark_2023
CSVs and scripts from [Azevedo, Lesser, Phelps, Mark et. al., 2023](url), as well as a space to host json states of neuroglancer views linked in the paper.

## Synapse comparisons for Figure 2 and Extended Data Figure 3
main.ipynb provides a script to reproduce the comparisons between manually annotated synapses and synapse locations that are automatically predicted. This script imports other modules located in the folder, but also depends on [caveclient](https://caveclient.readthedocs.io/en/latest/index.html), [cloudvolume](https://github.com/seung-lab/cloud-volume) and [fanc](https://github.com/htem/FANC_auto_recon). Please see those repositories for information on downloading. 

The synapse location data can also be found in data/synapse_tables/ as both csv files and pickle files of pandas dataframes. Note, the position coordinates are 3-element lists, which map to strings in the csv files. 
