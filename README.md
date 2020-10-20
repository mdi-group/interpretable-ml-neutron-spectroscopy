# interpretable-ml-neutron-spectroscopy
A repository of code associated with the publication _Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_

Data associated with training the neural networks in this repo is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4088240.svg)](https://doi.org/10.5281/zenodo.4088240)
## Generating Data

The training data may also be generated using the code in the `data_generation` folder.
To use this, you will need to download and install the (beer-free) Matlab runtime version 2017b [in this page](https://www.mathworks.com/products/compiler/matlab-runtime.html) for your OS.

## Running the codes

There are different `conda` environments associated with the different codes:

* To run the uncertainty quntification netowrks you will need to load the `pytorch` environment in `environment-torch.yml`
* To run the discrimination and class activation map networksyou will need to load the `tensorflow` environment in `environment-tf.yml`

## Notebook examples

The notebook examples in the `duq` and `interpret` directories load pre-trained models and apply them to experimental data, so that you can re-create the results from the paper without re-training the networks. The saved weights are too large for this *GitHub* repository, but are available in the associated data repository [**10.5281/zenodo.4088240**](https://zenodo.org/deposit/4088240) in the file `model-weights.tgz`. Once this file is untarred and unzipped, weights files corresponding to those in the notebooks will be present.   

To run the `duq` notebook you should launch a `jupyter` notebook in the `conda` environment described in `environment-torch.yml`
```
conda env create -f environment_torch.yml -n duq
conda activate duq
jupyter notebook
```

To run the `interpret` notebook you should launch a `jupyter` notebook in the `conda` environment described in `environment-tf.yml`
```
conda env create -f environment_tf.yml -n interpret
conda activate interpret
jupyter notebook
```

## Using Docker

You can also use the provided Dockerfile to build a Docker container to run the codes.

```
docker build -t ml_ins https://github.com/keeeto/interpretable-ml-neutron-spectroscopy
docker run --rm -ti -p 8888:8888 -p 8889:8889 ml_ins /bin/bash
```

This will put you into a command prompt. To run the data generation:

```
conda activate data
cd ~/interpretable-ml-neutron-spectroscopy/data_generation/resolution && python generate_goodenough_resolution.py
cd ~/interpretable-ml-neutron-spectroscopy/data_generation/resolution && python generate_dimer_resolution.py
cd ~/interpretable-ml-neutron-spectroscopy/data_generation/brille/goodenough && bash runjobgoodenough
cd ~/interpretable-ml-neutron-spectroscopy/data_generation/brille/dimer && bash runjobdimer
```

To run the notebooks

```
conda activate duq
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root &
conda activate interpret
jupyter notebook --ip=0.0.0.0 --port=8889 --allow-root &
```

Then you can browse the DUQ notebooks at `http://localhost:8888` and the CAM notebooks at `http://localhost:8889`.
Please note the tokens printed on the command line output as you need these to log on.

You can also run these containers in Windows, as long as you have the Windows Subsystem for Linux version 2 (WSL2) in Windows 10.
Please see the [docker documentation](https://docs.docker.com/docker-for-windows/wsl/) for how to run Docker under Windows.
