    cd next
    ./crop-imgs/run.sh

    python run.py \
      --job-name ... \
      --imgs-path ...

# examples

## test1

    job_name=test1
    top=0.5 # percentage
    left=0.5 # percentage
    crop_width=512
    crop_height=512
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

## test2

    job_name=test2
    top=0.5 # percentage
    left=0.2 # percentage
    crop_width=1024
    crop_height=1024
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
