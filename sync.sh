dir="combine-imgs"
gcloud compute scp ./$dir gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

dir="crop-imgs"
gcloud compute scp ./$dir gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

dir="face2landmarks"
gcloud compute scp ./$dir gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

dir="pix2pix-infer"
gcloud compute scp ./$dir gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

dir="pix2pix-trainer"
gcloud compute scp ./$dir gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

dir="video"
gcloud compute scp ./$dir gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

file="build.sh"
gcloud compute scp ./$file gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

file="infer.sh"
gcloud compute scp ./$file gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

file="sync.sh"
gcloud compute scp ./$file gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c

file="train.sh"
gcloud compute scp ./$file gpu:/home/next \
  --compress \
  --recurse \
  --project magggenta-176803 \
  --zone us-central1-c