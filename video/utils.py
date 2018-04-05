import subprocess


def call(cmd):
    print '$', cmd
    return subprocess.check_output(['bash', '-c', cmd])


def get_fps(video_path):
    cmd = 'ffprobe \
        -v 0 -of csv=p=0 \
        -select_streams 0 \
        -show_entries stream=r_frame_rate \
        {}'.format(video_path)
    output = call(cmd)
    return output[:-1]


def video_to_imgs(video_path, fps, output_pattern):
    cmd = 'ffmpeg \
        -i {} \
        -framerate {} \
        {}'.format(video_path, fps, output_pattern)
    call(cmd)


def imgs_to_video(input_pattern, fps, audio_path, output_video_path):
    # yuv420p: sets the pixel format to something QuickTime can read
    cmd = 'ffmpeg \
        -framerate {} \
        -i {} \
        -i {} \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -c:a copy \
        {}'.format(fps, input_pattern, audio_path, output_video_path)
    call(cmd)


def extract_audio(video_path, audio_path):
    # vn: for no video
    # copy: use same audio stream that's already in there
    cmd = 'ffmpeg \
        -i {} \
        -vn \
        -acodec copy \
        {}'.format(video_path, audio_path)
    call(cmd)
