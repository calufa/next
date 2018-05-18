FROM tensorflow/tensorflow:1.4.1-gpu

RUN apt-get update
RUN apt-get install screen
RUN pip install glob2==0.6

RUN mkdir /files
RUN mkdir /service
WORKDIR /service