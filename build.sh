service="combine-imgs"
docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

service="crop-imgs"
docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

service="face2landmarks"
docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

service="pix2pix-infer"
nvidia-docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

service="pix2pix-trainer"
nvidia-docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}

service="video"
docker build \
  -t ${service} \
  -f ${service}/Dockerfile \
  ${service}
