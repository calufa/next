import argparse

from utils import get_fps
from utils import imgs_to_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path')
    parser.add_argument('--imgs-path')
    parser.add_argument('--audio-path')
    parser.add_argument('--output-video-path')
    parser.add_argument('--file-pattern', default='%010d.png')
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print '{}={}'.format(k, v)

    video_path = args.video_path
    imgs_path = args.imgs_path
    audio_path = args.audio_path
    output_video_path = args.output_video_path
    file_pattern = args.file_pattern

    # get frames per second
    fps = get_fps(video_path)

    # create input pattern
    input_pattern = '{}/{}'.format(imgs_path, file_pattern)

    # convert seq of imgs into convert video
    imgs_to_video(input_pattern, fps, audio_path, output_video_path)
