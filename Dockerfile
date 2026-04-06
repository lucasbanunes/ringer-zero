FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get clean
# It takes a while to install this, implemented separately for caching
# RUN apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended git && apt-get clean
RUN apt-get install -y wget && apt-get clean


WORKDIR /root
COPY requirements.txt requirements.txt
# Root only supports conda or binary installation
# We install it with conda and the rest with pip
# to have the most recent packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py312_26.1.1-1-Linux-x86_64.sh -O /tmp/miniconda3.sh
RUN bash /tmp/miniconda3.sh -b -p /opt/conda && \
    rm /tmp/miniconda3.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda config --set auto_activate_base false && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda create -c conda-forge --name ringer-zero root==6.36.06 python==3.12.12 -y && \
    /opt/conda/bin/conda run -n ringer-zero --live-stream pip install -r requirements.txt && \
    rm requirements.txt
ENV PATH="/opt/conda/bin:${PATH}"
