$ cd next
$ ./pix2pix-trainer/run.sh

python run.py \
  --job-name ... \
  --imgs-path ...png \
  --max-epochs ...

# example
# 400 epochs = 15 hours
job_name=...
epochs=400
screen bash -c "python run.py \
  --job-name ${job_name} \
  --imgs-path /files/_combine-imgs/${job_name} \
  --max-epochs=${epochs} \
  > run.log"