# sync all
gcloud compute scp ./ gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

# sync /files
gcloud compute scp ./files gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c





screen python pix2pix.py \
  --mode train \
  --output_dir output \
  --max_epochs 800 \
  --input_dir B-A \
  --which_direction AtoB

gcloud compute --project "magggenta-176803" ssh --zone "us-central1-c" "gpu"




gcloud compute scp gpu:/home/pix2pix-tensorflow/eval-output/ ./ \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c




gcloud compute scp ./pix2pix2.py gpu:/home/pix2pix-tensorflow/pix2pix2.py \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

python pix2pix2.py \
  --output_dir output \
  --max_epochs 10


rm -rf eval-output && mkdir eval-output
python pix2pix.py \
  --mode test \
  --output_dir eval-output \
  --input_dir eval \
  --checkpoint output




gcloud compute scp gpu:/home/pix2pix-tensorflow/A-B ./ \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c


python tools/process.py \
  --input_dir A \
  --b_dir B \
  --operation combine \
  --output_dir A-B