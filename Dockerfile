# Sets up a Linux environment to run calculations 
# for machine learning of inelastic neutron scattering data

# Use Matlab runtime image
FROM demartis/matlab-runtime:R2020a

MAINTAINER Duc Le <duc.le@stfc.ac.uk>

ENV DEBIAN_FRONTEND noninteractive

# Need to use bash otherwise cannot activate Conda environment
SHELL ["/bin/bash", "-c"]

# Installs everything in one layer
RUN apt-get -q update \
    && apt-get install -q -y --no-install-recommends \
         xorg \
         unzip \
         wget \
         curl \
         bzip2 \
         ca-certificates \
         git \
         libgomp1 \
         build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /usr/local/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && /usr/local/miniconda3/bin/conda init bash \
    && source /root/.bashrc \
    && conda create -n data python=3.6 numpy \
    && git clone https://github.com/keeeto/interpretable-ml-neutron-spectroscopy \
    && conda env create -f interpretable-ml-neutron-spectroscopy/environment_torch.yml -n duq \
    && conda env create -f interpretable-ml-neutron-spectroscopy/environment_tf.yml -n interpret \
    && conda activate data \
    && pip install brille

# Configure environment variables for MCR
ENV LD_LIBRARY_PATH /opt/mcr/v98/runtime/glnxa64:/opt/mcr/v98/bin/glnxa64:/opt/mcr/v98/sys/os/glnxa64
