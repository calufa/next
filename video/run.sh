service_name="next-video"
service_dir="video"

docker build \
  -t ${service_name} \
  -f ${service_dir}/Dockerfile \
  ${service_dir}

docker run \
  -v $(pwd)/${service_dir}:/service \
  -v $(pwd)/files:/files \
  -it ${service_name} \
  bash
