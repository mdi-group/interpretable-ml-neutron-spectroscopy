# Sets up a Linux environment to generate spin-wave data for training
# for machine learning of inelastic neutron scattering data

# Use Matlab runtime image
FROM continuumio/anaconda3:2020.02

MAINTAINER Duc Le <duc.le@stfc.ac.uk>

ENV DEBIAN_FRONTEND noninteractive

# Need to use bash otherwise cannot activate Conda environment
SHELL ["/bin/bash", "-c"]

# Installs git repo and data in one layer to be used between this and the CAM env
RUN apt-get -q update \
    && apt-get install -q -y --no-install-recommends \
         git \
         build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/keeeto/interpretable-ml-neutron-spectroscopy \
    && cd interpretable-ml-neutron-spectroscopy \
    && wget https://zenodo.org/record/4088240/files/model-weights.tar.gz?download=1 -O model-weights.tar.gz \
    && tar zxvf model-weights.tar.gz \
    && rm -f model-weights.tar.gz

RUN \
    conda env create -f interpretable-ml-neutron-spectroscopy/environment_torch.yml -n duq \
    && echo "conda activate duq" >> /root/.bashrc

RUN \
    mkdir -p /root/.jupyter \
    && echo "{ \"NotebookApp\": { \"password\": \"sha1:e7d0488641d5:dae8affbad5b77a0242d16dcd97d4a93a2fdbd0a\" } }" > /root/.jupyter/jupyter_notebook_config.json \
    && printf "#\x21/bin/bash\nsource /root/.bashrc\njupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root\n" > /root/run_notebook.sh \
    && chmod 755 /root/run_notebook.sh

EXPOSE 8888
WORKDIR /interpretable-ml-neutron-spectroscopy
ENTRYPOINT ["/root/run_notebook.sh"]

# Configure environment variables for MCR
ENV LD_LIBRARY_PATH /opt/mcr/v98/runtime/glnxa64:/opt/mcr/v98/bin/glnxa64:/opt/mcr/v98/sys/os/glnxa64
