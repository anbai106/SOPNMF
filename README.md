# `pyOPNMF`
**pyOPNMF** is the fast python implementation of the Orthogonal Projection Non-negative Matrix Factorization: [brainparts](https://github.com/asotiras/brainparts).Compared to the matlab implementation, pyOPNMF is much computationally faster with multi-threads implementation.

## Installation
[Ananconda](https://www.anaconda.com/products/individual) allows you to install, run and update python package and their dependencies. We highly recommend the users to install **Anancond3** on your machine.
After installing Anaconda3, there are three choices to use pyOPNMF.
### Use pyOPNMF as a python package
We recommend the users to use Conda virtual environment:
```
1) conda create --name pyOPNMF python=3.6
```
Activate the virtual environment:
```
2) source activate pyOPNMF
```
Install other python package dependencies (go to the root folder of pyOPNMF):
```
3) pip install -r requirements.txt
```
Finally, we need install pyOPNMF from PyPi (**Not yet avaible now**):
```
3) pip install pyOPNMF==1.0.0
```

### Use pyOPNMF from commandline:
To come soon.

### Use pyOPNMF as a developer version (**Recommended now**):
```
python -m pip install git+https://github.com/anbai106/pyOPNMF.git
```

## participant tsv
pyOPNMF requires the input (*participant_tsv*) to be a specific structure inspired by [BIDS](https://bids.neuroimaging.io/). The 3 columns in the tsv are **participant_id**, **session_id** and **path**, respectively.

Example for participant tsv:
```
participant_id    session_id    diagnosis
sub-CLNC0001      ses-M00      absolute_path    
sub-CLNC0002      ses-M00      absolute_path
sub-CLNC0003      ses-M00      absolute_path
sub-CLNC0004      ses-M00      absolute_path
sub-CLNC0005      ses-M00      absolute_path
sub-CLNC0006      ses-M00      absolute_path
sub-CLNC0007      ses-M00      absolute_path
sub-CLNC0008      ses-M00      absolute_path
```

## Examples for usage

### First, training pyOPNMF:
```
from pyOPNMF.opnmf import train
participant_tsv="path_to_participant_tsv"
output_dir = "path_output_dir"
num_component_min = 2
num_component_max = 60
n_threads = 8
verbose = True
train(participant_tsv, output_dir, num_component_min, num_component_max, n_threads=n_threads, verbose=verbose)
```

### Second, applying the trained model to the training data for post-analyses:
```
from pyOPNMF.opnmf_post import apply_to_training
output_dir = "path_output_dir"
apply_to_training(output_dir, verbose=True)
```

### Last, you may also apply the trained model to unseen test data:
```
from pyOPNMF.opnmf_post import apply_to_test
participant_tsv="path_to_participant_tsv"
output_dir = "path_output_dir"
apply_to_test(participant_tsv, output_dir, verbose=True)
```

## Citing this work
> Sotiras, A., Resnick, S.M. and Davatzikos, C., 2015. **Finding imaging patterns of structural covariance via non-negative matrix factorization**. Neuroimage, 108, pp.1-16. [doi:10.1016/j.neuroimage.2014.11.045](https://www.sciencedirect.com/science/article/pii/S1053811914009756?via%3Dihub)

> Wen, J., Varol, E., Ganesh, C., Sotiras, A., Davatzikos, C., 2020. **MAGIC: Multi-scale Heterogeneity Analysis andClustering for Brain Diseases**. MICCAI 2020.
