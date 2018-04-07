$ cd next
$ ./pix2pix/run.sh

python run.py \
  --job-name ... \
  --imgs-path ...png

# 400 epochs = 15 hours
screen bash -c "python run.py \
  --job-name test1 \
  --imgs-path /files/_combine-imgs/test1 \
  --max-epochs=400 \
  > run.log"