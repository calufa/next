$ gcloud auth login
$ gcloud compute --project "magggenta-176803" ssh --zone "us-central1-c" "gpu"

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

# sync /files from local
DIR=files
gcloud compute scp ./$DIR gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

# sync /files from server
DIR=files
gcloud compute scp gpu:/home/next/${DIR} ./ \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

