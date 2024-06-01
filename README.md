This repository contains the code for experiments presented in:<br>
`Pedestrian intention prediction in Adverse Weather
Conditions with Spiking Neural Networks and
Dynamic Vision Sensors` <br>

__The work is this project is still in progress. All files are presented as is and may contain bugs. Comments, issues and contributions are welcome.__

## Overview

All experiments are meant to be configured via `config.yml` file. To get more information about the configuration file, please refer to the `config.yml` file in the root directory and the comments there.

## Datasets
The simulation dataset that is introduced in this work is available under the DOI: `10.5281/zenodo.11409259`. The dataset is not included in this repository and should be downloaded separately. <br>
Subset of JAAD dataset is used in this work, which can be downloaded [here](https://drive.google.com/drive/folders/1ZJmSg6jbROHoE6bPScIh1yknD2J4qzdz?usp=drive_link). For full dataset, please refer to the [JAAD website](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

## Experiments involving energy usage monitoring
Experiments measuring the number of synaptic operations were ran using the `syops` package available at `https://github.com/iCGY96/syops-counter`. The package is not included in this repository and should be installed separately.
Note that in order to run the experiments, manual changes were made in the code of the package. For further details, please contact the authors.


