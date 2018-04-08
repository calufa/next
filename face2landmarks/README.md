    cd next
    ./face2landmarks/run.sh

    python run.py \
      --job-name ... \
      --imgs-path /files/...

# example

    job_name=test1
    python run.py \
      --job-name ${job_name} \
      --imgs-path /files/_video/${job_name}