FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#python
RUN apt-get update \
  && apt-get install -y \
  apt-utils \
  python3-pip \
  python3-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  locales

# Library manager
RUN python3 -m pip install --upgrade pip
RUN pip3 install bencode-python3
RUN apt-get update \
  && apt-get install -y wget
RUN apt-get install -y gdebi-core

# python libraries
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip wheel
RUN pip3 install -r requirements.txt

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf \
    locale-gen en_US.UTF-8

#clean the package cache
RUN rm -rf /var/lib/apt/lists/*

#SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash
ENV AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
ENV AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
#ENV MLFLOW_S3_ENDPOINT_URL=http://nginx:80
ENV MLFLOW_S3_ENDPOINT_URL=http://s3server:9000
ENV MLFLOW_TRACKING_URI=http://nginx:80


WORKDIR /notebooks
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0","--NotebookApp.token='deepikerlan'","--allow-root"]
#jupyter notebook --ip 0.0.0.0 --NotebookApp.token='deepikerlan' --allow-root

