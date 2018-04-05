import argparse
import os

from utils import get_fps
from utils import video_to_imgs


if __name__ == '__main__':
    output_path = '/files/_video/imgs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path')
    parser.add_argument('--job-name')
    parser.add_argument('--output-path', default=output_path)
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print '{}={}'.format(k, v)

    video_path = args.video_path
    job_name = args.job_name
    output_path = args.output_path

    # get frames per second
    fps = get_fps(video_path)
    # create job output directory
    output_path = '{}/{}'.format(output_path, job_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # create output pattern
    output_pattern = '{}/%010d.png'.format(output_path)
    # convert video into seq of imgs
    video_to_imgs(video_path, fps, output_pattern)
