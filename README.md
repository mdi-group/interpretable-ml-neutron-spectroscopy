# interpretable-ml-neutron-spectroscopy
A repository of code associated with the publication _Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_

Data associated with training the neural networks in this repo is available at **10.5281/zenodo.4088240**

## Generating Data

The training data may also be generated using the code in the `data_generation` folder.
To use this, you will need to download and install the (beer-free) Matlab runtime version 2017b [in this page](https://www.mathworks.com/products/compiler/matlab-runtime.html) for your OS.

## Running the codes

There are different `conda` environments associated with the different codes:

* To run the uncertainty quntification netowrks you will need to load the `pytorch` environment in `environment-torch.yml`
* To run the discrimination and class activation map networksyou will need to load the `tensorflow` environment in `environment-tf.yml`

## Notebook examples

The notebook examples in the `duq` and `interpret` directories load pre-trained models and apply them to experimental data, so that you can re-create the results from the paper without re-training the networks. The saved weights are too large for this *GitHub* repository, but are available in the associated data repository **10.5281/zenodo.4088240** in the file `model-weights.tgz`. Once this file is untarred and unzipped, weights files corresponding to those in the notebooks will be present.   

To run the `duq` notebook you should launch a `jupyter` notebook in the `conda` environment described in `environment-torch.yml`
```
conda env create -f environment-torch.yml -n duq
conda activate duq
jupyter notebook
```

To run the `interpret` notebook you should launch a `jupyter` notebook in the `conda` environment described in `environment-tf.yml`
```
conda env create -f environment-tf.yml -n interpret
conda activate interpret
jupyter notebook
```
