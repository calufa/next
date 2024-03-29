job_name=test7
video_path=/files/${job_name}.mp4
# crop settings
top=0.35 # percentage
left=0.5 # percentage
crop_width=600
crop_height=600
resize_width=256
resize_height=256
# landmarks
landmarks='left_eye right_eye outer_lip inner_lip'
# training
epochs=40

echo '> build docker imgs'
./build.sh

echo '> video to imgs'
docker run \
  -v $(pwd)/video:/service \
  -v $(pwd)/files:/files \
  -it video \
  python video_to_imgs.py \
    --job-name ${job_name} \
    --video-path ${video_path}

echo '> extract facial landmarks'
docker run \
  -v $(pwd)/face2landmarks:/service \
  -v $(pwd)/files:/files \
  -it face2landmarks \
  python run.py \
    --job-name ${job_name} \
    --imgs-path /files/_video/${job_name} \
    --landmarks ${landmarks}

echo '> crop imgs'
docker run \
  -v $(pwd)/crop-imgs:/service \
  -v $(pwd)/files:/files \
  -it crop-imgs \
  python run.py \
    --job-name ${job_name}-A \
    --imgs-path /files/_face2landmarks/${job_name} \
    --top ${top} \
    --left ${left} \
    --crop-width ${crop_width} \
    --crop-height ${crop_height} \
    --resize-width ${resize_width} \
    --resize-height ${resize_height}
docker run \
  -v $(pwd)/crop-imgs:/service \
  -v $(pwd)/files:/files \
  -it crop-imgs \
  python run.py \
    --job-name ${job_name}-B \
    --imgs-path /files/_video/${job_name} \
    --top ${top} \
    --left ${left} \
    --crop-width ${crop_width} \
    --crop-height ${crop_height} \
    --resize-width ${resize_width} \
    --resize-height ${resize_height}

echo '> combine imgs'
docker run \
  -v $(pwd)/combine-imgs:/service \
  -v $(pwd)/files:/files \
  -it combine-imgs \
  python run.py \
    --job-name ${job_name} \
    --A-path /files/_crop-imgs/${job_name}-A \
    --B-path /files/_crop-imgs/${job_name}-B

echo '> train model'
nvidia-docker run \
  -v $(pwd)/pix2pix-trainer:/service \
  -v $(pwd)/files:/files \
  -it pix2pix-trainer \
  screen bash -c "python no_flip.py \
    --job-name ${job_name} \
    --imgs-path /files/_combine-imgs/${job_name} \
    --max-epochs=${epochs} \
    &> run.log"
