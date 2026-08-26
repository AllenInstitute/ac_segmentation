FROM continuumio/miniconda3:23.10.0-1 as ac_segmentation

SHELL ["/bin/bash", "-c"]

# Update conda and create clean environment
RUN conda update -y conda && \
    conda create -y -n ac -c conda-forge python=3.10 gcc=12.3.0 pip && \
    conda clean -a

COPY . /ac_segmentation

# Install /ac_segmentation
WORKDIR /ac_segmentation
RUN source activate ac && \
    pip install . && \
    conda clean -a

ENTRYPOINT ["/bin/bash", "/ac_segmentation/entrypoint.sh"]
WORKDIR /

