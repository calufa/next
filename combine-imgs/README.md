    cd next
    ./combine-imgs/run.sh

    python run.py \
      --job-name ... \
      --A-path ... \
      --B-path ...

# example

## test1

    job_name=test1
    python run.py \
      --job-name ${job_name} \
      --A-path /files/_crop-imgs/${job_name}-A \
      --B-path /files/_crop-imgs/${job_name}-B

## test2

    job_name=test2
    python run.py \
      --job-name ${job_name} \
      --A-path /files/_crop-imgs/${job_name}-A \
      --B-path /files/_crop-imgs/${job_name}-B
