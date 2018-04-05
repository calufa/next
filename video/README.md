$ cd next
$ ./video/run.sh

python video_to_imgs.py \
  --video-path /files/...mp4 \
  --job-name ...

python extract_audio.py \
  --video-path /files/...mp4 \
  --audio-path /files/...aac

python imgs_to_video.py \
  --video-path /files/...mp4 \
  --imgs-path /files/... \
  --audio-path /files/...aac \
  --output-video-path /files/...mp4
