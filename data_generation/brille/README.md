# Generating training dataset with Monte Carlo resolution

This folder contains the compiled Matlab executable code and support Python files to generate the training datasets using the Monte Carlo resolution-convolution method.

A (Linux only) executable called `model_eval` is used to run the calculation. 
However, this 421MB file is too large to be stored in this git repository.
Instead it is available at: [10.5281/zenodo.4088240](https://doi.org/10.5281/zenodo.4088240).
Please download that archive and place the `model_eval` file in this folder.
Like the Python modules in the `resolution` folder it also requires the [Matlab Compiler Runtime (MCR) version 2017b](https://www.mathworks.com/products/compiler/matlab-runtime.html).
In addition, to run this executable you also have to have the `brille` Python module installed (using e.g. `pip install brille`).
Finally, the executable has hard coded in the path `/usr/bin/python3` as the python interpreter.
If your system Python is not located there please create a symlink to it.

The are subfolders for the `goodenough` and `dimer` models, each with a `runjob` script to be run using `sbatch --array=0-199 runjob`.
Each `runjob` script runs a separate `runjob[goodenough|dimer]` script to run sets of 10 calculations each iteration.

In order to know the exact pixel coordinates in order to perform the Monte Carlo resolution convolution,
these scripts require the actual measured data files which are available at the [Zenodo archive](https://doi.org/10.5281/zenodo.4088240) noted above.
Please download these and place in the `measured_data` folder above this subfolder before the computation (or change the symlinks in the `[goodenough|dimer]` folders).

The generated data sets used in the work _Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_
are also available at the same [Zenodo repository](https://doi.org/10.5281/zenodo.4088240).
