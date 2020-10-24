# Convolution neural networks

This folder contains the CNN models for the work _Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_

The `classify`, `vae` and `interpret` folders contains Python codes based on `tensorflow`
that defines a convolution neural network for classification of the synthetic data (in `classify`),
a variational autoencoder to denoise the data (in `vae`)
or defines a CNN classifier based on cleaned data with class activation map visualisation (in `interpret`).
The `interpret` folder also has a Jupyter notebook for the visualisation.

The `duq` folder contains Python codes based on PyTorch to define a convolution neural network classifier
with deterministic uncertainty quantification.
This also includes a Jupyter notebook.

# Docker

The `duq` and `interpret` folders also contain `Dockerfile`s to build Docker containers for the
PyTorch and tensorflow environments respectively, using:

```
docker build -t ml_ins_duq https://raw.githubusercontent.com/keeeto/interpretable-ml-neutron-spectroscopy/main/models/duq/Dockerfile
docker build -t ml_ins_interpret https://raw.githubusercontent.com/keeeto/interpretable-ml-neutron-spectroscopy/main/models/interpret/Dockerfile
```

These containers are available on the Docker Hub and can be pulled using:

```
docker pull mducle/ml_ins_duq
docker pull mducle/ml_ins_interpret
```

When run they will start a Jupyter notebook server so the respective notebooks can be viewed.
