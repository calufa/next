FROM python:2.7-slim

RUN apt-get -y update
RUN apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libatlas-dev \
    libgtk2.0-dev \
    libboost-all-dev

RUN pip install dlib==19.8
RUN pip install opencv-python==3.4.0.12
RUN pip install glob2==0.6

RUN mkdir /files
RUN mkdir /service
WORKDIR /service
