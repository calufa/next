import argparse
import glob2
import os

from PIL import Image


if __name__ == '__main__':
    output_path = '/files/_combine-imgs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name')
    parser.add_argument('--A-path')
    parser.add_argument('--B-path')
    parser.add_argument('--output-path', default=output_path)
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print '{}={}'.format(k, v)

    job_name = args.job_name
    A_path = args.A_path
    B_path = args.B_path
    output_path = args.output_path

    # create job output directory
    output_path = '{}/{}'.format(output_path, job_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get and sort files
    A_pattern_path = '{}/*'.format(A_path)
    A_files = glob2.glob(A_pattern_path)
    A_files = sorted(A_files)

    B_pattern_path = '{}/*'.format(B_path)
    B_files = glob2.glob(B_pattern_path)
    B_files = sorted(B_files)

    for i, _ in enumerate(A_files):
        print i + 1, '/', len(A_files)

        A_file = A_files[i]
        A_file = Image.open(A_file)

        B_file = B_files[i]
        B_file = Image.open(B_file)

        # img.size returns tuple (width, height)
        total_width = A_file.size[0] + B_file.size[0]
        max_height = max((A_file.size[1], B_file.size[1]))

        # create new img
        combined_img = Image.new('RGB', (total_width, max_height))

        # combine imgs
        offset = 0
        for img in [A_file, B_file]:
            combined_img.paste(img, (offset, 0))
            offset += img.size[0]

        # save imgs
        file_name = '{}.png'.format(i)
        output_file = '{}/{}'.format(output_path, file_name)
        combined_img.save(output_file)
