service="video"

docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

docker run \
  -v $(pwd)/${service}:/service \
  -v $(pwd)/files:/files \
  -it ${service} \
  bash
