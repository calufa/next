FROM alpine

RUN apk update
RUN apk add \
  bash \
  ffmpeg \
  python-dev

RUN mkdir /files
RUN mkdir /service
WORKDIR /service
