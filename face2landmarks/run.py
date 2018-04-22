import argparse
import cv2
import dlib
import glob2
import numpy as np
import os

from os.path import basename


def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    img = np.zeros(img.shape, np.uint8)
    color = (255, 255, 255)
    thickness = 3

    for face in faces:
        landmarks = predictor(gray, face).parts()
        landmarks = [[p.x, p.y] for p in landmarks]

        if 'jaw' in args.landmarks:
            jaw = reshape_for_polyline(landmarks[0:17])
            cv2.polylines(img, [jaw], False, color, thickness)

        if 'left_eyebrow' in args.landmarks:
            left_eyebrow = reshape_for_polyline(landmarks[22:27])
            cv2.polylines(img, [left_eyebrow], False, color, thickness)

        if 'right_eyebrow' in args.landmarks:
            right_eyebrow = reshape_for_polyline(landmarks[17:22])
            cv2.polylines(img, [right_eyebrow], False, color, thickness)

        if 'nose_bridge' in args.landmarks:
            nose_bridge = reshape_for_polyline(landmarks[27:31])
            cv2.polylines(img, [nose_bridge], False, color, thickness)

        if 'lower_nose' in args.landmarks:
            lower_nose = reshape_for_polyline(landmarks[30:35])
            cv2.polylines(img, [lower_nose], True, color, thickness)

        if 'left_eye' in args.landmarks:
            left_eye = reshape_for_polyline(landmarks[42:48])
            cv2.polylines(img, [left_eye], True, color, thickness)

        if 'right_eye' in args.landmarks:
            right_eye = reshape_for_polyline(landmarks[36:42])
            cv2.polylines(img, [right_eye], True, color, thickness)

        if 'outer_lip' in args.landmarks:
            outer_lip = reshape_for_polyline(landmarks[48:60])
            cv2.polylines(img, [outer_lip], True, color, thickness)

        if 'inner_lip' in args.landmarks:
            inner_lip = reshape_for_polyline(landmarks[60:68])
            cv2.polylines(img, [inner_lip], True, color, thickness)

    return img


def reshape_for_polyline(array):
    # reshape image so that it works with polyline
    return np.array(array, np.int32).reshape((-1, 1, 2))


if __name__ == '__main__':
    output_path = '/files/_face2landmarks'
    landmarks = ['jaw',
                 'left_eyebrow', 'right_eyebrow',
                 'nose_bridge', 'lower_nose',
                 'left_eye', 'right_eye',
                 'outer_lip', 'inner_lip']

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name')
    parser.add_argument('--imgs-path')
    parser.add_argument('--output-path', default=output_path)
    parser.add_argument('--landmarks', default=landmarks, nargs='+')
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print '{}={}'.format(k, v)

    output_path = args.output_path

    # load vision models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # create job output directory
    output_path = '{}/{}'.format(output_path, args.job_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get and sort files
    pattern_path = '{}/*'.format(args.imgs_path)
    files = glob2.glob(pattern_path)
    files = sorted(files)

    for i, file_path in enumerate(files):
        print i + 1, '/', len(files), file_path

        file_name = basename(file_path)
        output_file = '{}/{}'.format(output_path, file_name)
        img = cv2.imread(file_path)
        img = get_landmarks(img)
        cv2.imwrite(output_file, img)
