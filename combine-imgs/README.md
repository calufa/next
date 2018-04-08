    cd next
    ./combine-imgs/run.sh

    python run.py \
      --job-name ... \
      --A-path ... \
      --B-path ...

# example

    $job_name=...
    python run.py \
      --job-name ${job_name} \
      --A-path /files/_crop-imgs/${job_name}-A \
      --B-path /files/_crop-imgs/${job_name}-B