# TRAIN

    job_name=test4
    video_path=/files/${job_name}.mp4
    docker run \
      -v $(pwd)/video:/service \
      -v $(pwd)/files:/files \
      -it video \
      python video_to_imgs.py \
        --job-name ${job_name} \
        --video-path ${video_path}

    docker run \
      -v $(pwd)/face2landmarks:/service \
      -v $(pwd)/files:/files \
      -it face2landmarks \
      python run.py \
        --job-name ${job_name} \
        --imgs-path /files/_video/${job_name} \
        --landmarks left_eye right_eye outer_lip inner_lip

    top=0.5 # percentage
    left=0.5 # percentage
    crop_width=720
    crop_height=720
    resize_width=256
    resize_height=256
    docker run \
      -v $(pwd)/crop-imgs:/service \
      -v $(pwd)/files:/files \
      -it crop-imgs \
      python run.py \
        --job-name ${job_name}-A \
        --imgs-path /files/_face2landmarks/${job_name} \
        --top ${top} \
        --left ${left} \
        --crop-width ${crop_width} \
        --crop-height ${crop_height} \
        --resize-width ${resize_width} \
        --resize-height ${resize_height} && \
    docker run \
      -v $(pwd)/crop-imgs:/service \
      -v $(pwd)/files:/files \
      -it crop-imgs \
      python run.py \
        --job-name ${job_name}-B \
        --imgs-path /files/_video/${job_name} \
        --top ${top} \
        --left ${left} \
        --crop-width ${crop_width} \
        --crop-height ${crop_height} \
        --resize-width ${resize_width} \
        --resize-height ${resize_height}

    docker run \
      -v $(pwd)/combine-imgs:/service \
      -v $(pwd)/files:/files \
      -it combine-imgs \
      python run.py \
        --job-name ${job_name} \
        --A-path /files/_crop-imgs/${job_name}-A \
        --B-path /files/_crop-imgs/${job_name}-B

    ./pix2pix-trainer/run.sh

    job_name=test7
    epochs=400
    screen bash -c "python run.py \
      --job-name ${job_name} \
      --imgs-path /files/_combine-imgs/${job_name} \
      --max-epochs=${epochs} \
      &> run.log"