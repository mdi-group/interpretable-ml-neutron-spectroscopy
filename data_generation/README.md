# Generating Training Data

This folder contains the code to generate the training datasets.

The folder `brille` contains "compiled" Matlab code to generate the Monte Carlo resolution-convolved data.

The folder `resolution` contains code to generate the pre-computed Gaussian-approximation resolution-convolved data.

The `src` folder contains the Matlab source code.

# Docker

The `Dockerfile` can be used to build a Docker container to carry out these calculations:

```
docker build -t ml_ins_data_generation docker build -t ml_ins_data_generation https://raw.githubusercontent.com/keeeto/interpretable-ml-neutron-spectroscopy/main/data_generation/Dockerfile
```

This container is available on the Docker Hub and can be pulled using:

```
docker pull mducle/ml_ins_data_generation
```
