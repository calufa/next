FROM python:2.7-slim

RUN apt-get -y upgrade
RUN apt-get -y update
RUN apt-get install -y \
    git \
    cmake \
    build-essential \
    pkg-config \
    libatlas-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    python-numpy

RUN cd ~ && \
    mkdir -p dlib && \
    git clone https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python setup.py install --yes USE_AVX_INSTRUCTIONS

RUN pip install imutils
RUN pip install opencv-python
RUN pip install glob2

RUN mkdir /files
RUN mkdir /service
WORKDIR /service
