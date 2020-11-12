# interpretable-ml-neutron-spectroscopy
A repository of code associated with the publication [_Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_](https://arxiv.org/abs/2011.04584)

Data associated with training the neural networks in this repo is available at ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4270057.svg)]
## Generating Data

The training data may also be generated using the code in the `data_generation` folder.
To use this, you will need to download and install the (beer-free) Matlab runtime version 2017b [in this page](https://www.mathworks.com/products/compiler/matlab-runtime.html) for your OS.

## Running the codes

There are different `conda` environments associated with the different codes:

* To run the uncertainty quntification netowrks you will need to load the `pytorch` environment in `environment-torch.yml`
* To run the discrimination and class activation map networksyou will need to load the `tensorflow` environment in `environment-tf.yml`

## Notebook examples

The notebook examples in the `duq` and `interpret` directories load pre-trained models and apply them to experimental data, so that you can re-create the results from the paper without re-training the networks. The saved weights are too large for this *GitHub* repository, but are available in the associated data repository [**10.5281/zenodo.4088240**](https://zenodo.org/deposit/4088240) in the file `model-weights.tgz`. Once this file is untarred and unzipped, weights files corresponding to those in the notebooks will be present.   

To run the `duq` notebook you should launch a `jupyter` notebook in the `conda` environment described in `environment-torch.yml`. Alternatively you may wish to run the notebooks in a `docker` container, see the section below for directions.
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

You can also use Docker containers to run the codes.
There are three containers, one to run SpinW/Brille to generate the training data,
one to run the DUQ classifier and one to run the class activation maps.

For the training data generation:

```
docker pull mducle/ml_ins_data_generation
docker run -ti mducle/ml_ins_data_generation /bin/bash
```

This will put you into a command prompt. To run the data generation:

```
cd /interpretable-ml-neutron-spectroscopy/data_generation/resolution && python generate_goodenough_resolution.py
cd /interpretable-ml-neutron-spectroscopy/data_generation/resolution && python generate_dimer_resolution.py
cd /interpretable-ml-neutron-spectroscopy/data_generation/brille/goodenough && bash runjobgoodenough
cd /interpretable-ml-neutron-spectroscopy/data_generation/brille/dimer && bash runjobdimer
```

For the DUQ notebook:

```
docker pull mducle/ml_ins_duq
docker run -ti -p8888:8888 ml_ins_duq 
```

This will start the notebook.
You should then navigate to `http://localhost:8888/notebooks/models/duq/notebook/duq-publication.ipynb` to load the notebook.
The password is `pcsmo`.

For the class-activation map notebooks:

```
docker pull mducle/ml_ins_interpret
docker run -ti -p8889:8889 ml_ins_interpret
```

And navigate to `http://localhost:8889/notebooks/models/interpret/cam-publication.ipynb`.
