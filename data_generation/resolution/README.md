# Generating training dataset with precomputed resolution ellipsoids 

This folder contains the compiled Matlab code to run SpinW under a Python environment, and Python code to generate the training datasets using the precomputed resolution function.

The precomputed resolution ellipsoid covariance matrix is stored in the `PCSMO_ei*_resolution.txt` files.
The 12 columns are in order: `Qh`, `Qk`, `EN`, `shh`, `shk`, `shE`, `skh`, `skk`, `skE`, `sEh`, `sEk`, `sEE`.
Since the excitations are approximately non-dispersive in the `Ql` direction, we ignore it in the calculation.
(The measured data was integrated over this direction to improve the signal.)
The first 3 columns of the resolution file contains the coordinates where the resolution ellipsoid covariance matrix was calculated,
the momentum transfer along `Qh` and `Qk` in reciprocal lattice units (r.l.u.) and the energy transfer `EN` in meV
The next 9 columns contain elements of the 3x3 covariance matrix in mixed units of r.l.u-meV.
The file `resolution_function.py` contains code to read and interpolate the precomputed resolution.

The files `generate_[goodenough|dimer]_resolution.py` files are meant to be run from the commandline and generate a set of 10 data sets (400x240 pixel images) indexed by the first argument.
The file `runjob` is a batch script to run these Python files on a cluster, using the syntax `sbatch --array=0-199 runjob` (which would generate 2000 data sets).
Each dataset is saved as a `numpy` array file with extension `.npy`.

Once these datasets are generated, the `build_[goodenough|dimer]_dataset.py` files can be run from the commmandline to concatenate the individual datasets to a single `.npy` file.

The datasets generated from this processed and used in the work _Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_
is available at [10.5281/zenodo.4088240](https://doi.org/10.5281/zenodo.4088240).
