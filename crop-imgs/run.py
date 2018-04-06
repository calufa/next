import argparse
import cv2
import glob2
import os

from os.path import basename


def crop(img, top, left, crop_width, crop_height):
    height, width, channels = img.shape
    top = int((height * top) - (crop_height * top))
    left = int((width * left) - (crop_width * left))
    return img[top:top + crop_height, left:left + crop_width]


if __name__ == '__main__':
    output_path = '/files/_crop-imgs'
    top = 0.5
    left = 0.5
    crop_width = 512
    crop_height = 512
    resize_width = 256
    resize_height = 256

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name')
    parser.add_argument('--imgs-path')
    parser.add_argument('--output-path', default=output_path)
    parser.add_argument('--top', default=top)
    parser.add_argument('--left', default=left)
    parser.add_argument('--crop-width', default=crop_width)
    parser.add_argument('--crop-height', default=crop_height)
    parser.add_argument('--resize-width', default=resize_width)
    parser.add_argument('--resize-height', default=resize_height)
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print '{}={}'.format(k, v)

    job_name = args.job_name
    imgs_path = args.imgs_path
    output_path = args.output_path
    top = float(args.top)
    left = float(args.left)
    crop_width = args.crop_width
    crop_height = args.crop_height
    resize_width = args.resize_width
    resize_height = args.resize_height

    # create job output directory
    output_path = '{}/{}'.format(output_path, job_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get and sort files
    pattern_path = '{}/*'.format(imgs_path)
    files = glob2.glob(pattern_path)
    files = sorted(files)

    for i, file_path in enumerate(files):
        print i + 1, '/', len(files), file_path

        img = cv2.imread(file_path)
        img = crop(img, top, left, crop_width, crop_height)
        img = cv2.resize(img, (resize_width, resize_height))

        file_name = basename(file_path)
        output_file = '{}/{}'.format(output_path, file_name)
        cv2.imwrite(output_file, img)
