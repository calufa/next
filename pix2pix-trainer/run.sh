service="pix2pix-trainer"

nvidia-docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

nvidia-docker run \
  -v $(pwd)/${service}:/service \
  -v $(pwd)/files:/files \
  -it ${service} \
  bash
