# EVAL

./crop-imgs/run.sh

job_name=test3
top=0.55 # percentage
left=0.5 # percentage
crop_width=512
crop_height=512
resize_width=256
resize_height=256
python run.py \
  --job-name ${job_name}-A \
  --imgs-path /files/_face2landmarks/${job_name} \
  --top ${top} \
  --left ${left} \
  --crop-width ${crop_width} \
  --crop-height ${crop_height} \
  --resize-width ${resize_width} \
  --resize-height ${resize_height}
python run.py \
  --job-name ${job_name}-B \
  --imgs-path /files/_video/${job_name} \
  --top ${top} \
  --left ${left} \
  --crop-width ${crop_width} \
  --crop-height ${crop_height} \
  --resize-width ${resize_width} \
  --resize-height ${resize_height}




./combine-imgs/run.sh

job_name=test3
python run.py \
  --job-name ${job_name} \
  --A-path /files/_crop-imgs/${job_name}-A \
  --B-path /files/_crop-imgs/${job_name}-B


gcloud compute scp ./files/_combine-imgs/test3 gpu:/home/next/files/_combine-imgs/  --compress \
--recurse \
--project magggenta-176803 \
--zone us-central1-c



./pix2pix-infer/run.sh

model_name=test1
job_name=test3
output_name=test3B
checkpoint=/files/_pix2pix-trainer/${model_name}
input_dir=/files/_combine-imgs/${job_name}
output_dir=/files/_pix2pix-infer/${output_name}
mkdir -p output_dir
python run.py \
  --mode test \
  --checkpoint ${checkpoint} \
  --input_dir  ${input_dir} \
  --output_dir ${output_dir}


gcloud compute scp gpu:/home/next/files/_pix2pix-infer/test3B ./files/_pix2pix-infer/   --compress \
--recurse \
--project magggenta-176803 \
--zone us-central1-c



./video/run.sh

job_name=test3
python extract_audio.py \
  --video-path /files/${job_name}.mp4 \
  --audio-path /files/${job_name}.aac

video_name=test3
job_name=test3B
audio_name=test3
output_video_name=test3B
video_path=/files/${video_name}.mp4
python imgs_to_video.py \
  --video-path ${video_path} \
  --imgs-path /files/_pix2pix-infer/${job_name}/images \
  --audio-path /files/${audio_name}.aac \
  --output-video-path /files/${output_video_name}-gen.mp4 \
  --file-pattern %d-outputs.png
