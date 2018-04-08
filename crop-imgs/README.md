    cd next
    ./crop-imgs/run.sh

    python run.py \
      --job-name ... \
      --imgs-path ...

# examples

    job_name=test1
    python run.py \
      --job-name ${job_name}-A \
      --imgs-path /files/_face2landmarks/${job_name}

    python run.py \
      --job-name ${job_name}-B \
      --imgs-path /files/_video/${job_name}