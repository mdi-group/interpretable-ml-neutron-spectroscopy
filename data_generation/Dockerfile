# Sets up a Linux environment to generate spin-wave data for training
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
         wget \
         curl \
         ca-certificates \
         git \
         libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/keeeto/interpretable-ml-neutron-spectroscopy \
    && cd interpretable-ml-neutron-spectroscopy \
    && wget https://zenodo.org/record/4088240/files/data_generation.tar.gz?download=1 -O data_generation.tar.gz \
    && tar zxvf data_generation.tar.gz \
    && rm -f data_generation.tar.gz \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /usr/local/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && /usr/local/miniconda3/bin/conda init bash \
    && source /root/.bashrc \
    && conda create -n data python=3.6 numpy \
    && conda activate data \
    && pip install brille \
    && echo "conda activate data" >> /root/.bashrc

# Configure environment variables for MCR
ENV LD_LIBRARY_PATH /opt/mcr/v98/runtime/glnxa64:/opt/mcr/v98/bin/glnxa64:/opt/mcr/v98/sys/os/glnxa64
