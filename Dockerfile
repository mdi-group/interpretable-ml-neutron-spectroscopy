# Sets up a Linux environment to run calculations 
# for machine learning of inelastic neutron scattering data

# Use official Python 3 images based on Debian.
FROM python:3.6-slim-stretch

LABEL maintainter="Duc Le <duc.le@stfc.ac.uk>"

ENV DEBIAN_FRONTEND noninteractive

# Installs required packages
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install numpy brille \
    && ln -s /usr/local/bin/python3 /usr/bin/python3

# Installs Matlab MCR (use one line to avoid multiple layers)
RUN mkdir /mcr-install \
    && mkdir /opt/mcr \
    && cd /mcr-install \
    && wget -nv https://ssd.mathworks.com/supportfiles/downloads/R2020a/Release/5/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2020a_Update_5_glnxa64.zip \
    && unzip MATLAB_Runtime_R2020a_Update_5_glnxa64.zip \
    && rm -f MATLAB_Runtime_R2020a_Update_5_glnxa64.zip \
    && ./install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent \
    && cd / \
    && rm -rf mcr-install

# Configure environment variables for MCR
ENV LD_LIBRARY_PATH /opt/mcr/v98/runtime/glnxa64:/opt/mcr/v98/bin/glnxa64:/opt/mcr/v98/sys/os/glnxa64
