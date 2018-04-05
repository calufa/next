import argparse

from utils import extract_audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path')
    parser.add_argument('--audio-path')
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print '{}={}'.format(k, v)

    video_path = args.video_path
    audio_path = args.audio_path

    extract_audio(video_path, audio_path)
