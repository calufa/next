service="pix2pix-infer"

nvidia-docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

nvidia-docker run \
  -v $(pwd)/${service}:/service \
  -v $(pwd)/files:/files \
  -it ${service} \
  bash
