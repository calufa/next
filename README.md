# sync all
gcloud compute scp ./ gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

# sync dir
DIR=pix2pix
gcloud compute scp ./$DIR gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

# sync /files
DIR=files
gcloud compute scp ./$DIR gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

