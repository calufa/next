# TRAIN

    ./video/run.sh

    job_name=test5
    video_path=/files/${job_name}.mp4
    python video_to_imgs.py \
      --job-name ${job_name} \
      --video-path ${video_path}

    ./face2landmarks/run.sh

    job_name=test5
    python run.py \
      --job-name ${job_name} \
      --imgs-path /files/_video/${job_name}

    ./crop-imgs/run.sh

    job_name=test5
    top=0.5 # percentage
    left=0.5 # percentage
    crop_width=720
    crop_height=720
    resize_width=256
    resize_height=256
    python run.py \
      --job-name ${job_name}-A \
      --imgs-path /files/_face2landmarks/${job_name} \
      --top ${top} \
      --left ${left} \
      --crop-width ${crop_width} \
      --crop-height ${crop_height} \
      --resize-width ${resize_width} \
      --resize-height ${resize_height}

    python run.py \
      --job-name ${job_name}-B \
      --imgs-path /files/_video/${job_name} \
      --top ${top} \
      --left ${left} \
      --crop-width ${crop_width} \
      --crop-height ${crop_height} \
      --resize-width ${resize_width} \
      --resize-height ${resize_height}

    ./combine-imgs/run.sh

    job_name=test5
    python run.py \
      --job-name ${job_name} \
      --A-path /files/_crop-imgs/${job_name}-A \
      --B-path /files/_crop-imgs/${job_name}-B

    ./pix2pix-trainer/run.sh

    job_name=test5
    epochs=400
    screen bash -c "python run.py \
      --job-name ${job_name} \
      --imgs-path /files/_combine-imgs/${job_name} \
      --max-epochs=${epochs} \
      &> run.log"