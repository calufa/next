    cd next
    ./video/run.sh

    python video_to_imgs.py \
      --job-name ... \
      --video-path /files/...mp4

    python extract_audio.py \
      --video-path /files/...mp4 \
      --audio-path /files/...aac

    python imgs_to_video.py \
      --video-path /files/...mp4 \
      --imgs-path /files/...png \
      --audio-path /files/...aac \
      --output-video-path /files/...mp4

# examples

## test video to imgs

    job_name=test...
    video_path=/files/${job_name}.mp4
    python video_to_imgs.py \
      --job-name ${job_name} \
      --video-path ${video_path}

## test extract audio

    job_name=test...
    python extract_audio.py \
      --video-path /files/${job_name}.mp4 \
      --audio-path /files/${job_name}.aac

## test image to video

    job_name=test...
    audio_name=test..
    video_path=/files/${job_name}.mp4
    python imgs_to_video.py \
      --video-path ${video_path} \
      --imgs-path /files/_pix2pix-infer/${job_name}/images \
      --audio-path /files/${audio_name}.aac \
      --output-video-path /files/${job_name}-gen.mp4 \
      --file-pattern %d-outputs.png

