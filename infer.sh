job_name=test9
model_name=test7
model_version=10.0
# crop settings
top=0.35 # percentage
left=0.5 # percentage
crop_width=600
crop_height=600
resize_width=256
resize_height=256
# landmarks
landmarks='left_eye right_eye outer_lip inner_lip'

echo '> build docker imgs'
./build.sh

echo '> video to imgs'
video_path=/files/${job_name}.mp4
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

echo '> infer'
checkpoint=/files/_pix2pix-trainer/${model_name}/${model_version}
input_dir=/files/_combine-imgs/${job_name}
output_dir=/files/_pix2pix-infer/${job_name}/${model_name}-${model_version}
mkdir -p ${PWD}/${output_dir}
nvidia-docker run \
  -v $(pwd)/pix2pix-infer:/service \
  -v $(pwd)/files:/files \
  -it pix2pix-infer \
  python run.py \
    --mode test \
    --checkpoint ${checkpoint} \
    --input_dir  ${input_dir} \
    --output_dir ${output_dir}

echo '> extract audio'
rm -f ${PWD}/files/${job_name}.aac
docker run \
  -v $(pwd)/video:/service \
  -v $(pwd)/files:/files \
  -it video \
  python extract_audio.py \
    --video-path /files/${job_name}.mp4 \
    --audio-path /files/${job_name}.aac

echo '> create video from infered imgs'
rm -f ${PWD}/files/${output_video_name}-gen.mp4
docker run \
  -v $(pwd)/video:/service \
  -v $(pwd)/files:/files \
  -it video \
  python imgs_to_video.py \
    --video-path ${video_path} \
    --imgs-path /files/_pix2pix-infer/${job_name}/${model_name}-${model_version}/images \
    --audio-path /files/${job_name}.aac \
    --output-video-path /files/${job_name}-${model_name}-${model_version}-gen.mp4 \
    --file-pattern %d-outputs.png
