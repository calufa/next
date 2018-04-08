from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import math
import numpy as np
import os
import random
import tensorflow as tf
import time


def load_samples(input_paths):
    with tf.name_scope('load_images'):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)

        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)

        raw_input = tf.image.decode_png(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1]  # [height, width, channels]
        inputs = preprocess(raw_input[:, :width // 2, :])
        targets = preprocess(raw_input[:, width // 2:, :])

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)

    with tf.name_scope('input_images'):
        input_images = random_flip_image(inputs, seed)

    with tf.name_scope('target_images'):
        target_images = random_flip_image(targets, seed)

    paths_batch, inputs_batch, targets_batch = tf.train.batch(
        [paths, input_images, target_images], batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

    return Samples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        steps_per_epoch=steps_per_epoch
    )


def random_flip_image(image, seed):
    i = tf.image.random_flip_left_right(image, seed=seed)
    # sets the shape of the tensor
    i = tf.image.crop_to_bounding_box(i, 0, 0, image_size, image_size)
    return i


def create_model(inputs, targets):
    with tf.variable_scope('generator'):
        out_channels = int(inputs.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator
    # one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope('discriminator_loss'):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope('generator_loss'):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope('discriminator_train'):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope('generator_train'):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope('encoder_1'):
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope('decoder_1'):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope('layer_1'):
        convolved = discrim_conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope('layer_%d' % (len(layers) + 1)):
            out_channels = ndf * min(2**(i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope('layer_%d' % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same', kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope('lrelu'):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def convert_image_dtype(image):
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


def train():
    # set seed
    seed = random.randint(0, 2**31 - 1)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load samples
    pattern_path = '{}/*'.format(imgs_path)
    input_paths = glob.glob(pattern_path)
    samples = load_samples(input_paths)
    print('samples count:', len(input_paths))

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(samples.inputs, samples.targets)
    inputs = deprocess(samples.inputs)
    targets = deprocess(samples.targets)
    outputs = deprocess(model.outputs)

    with tf.name_scope('predict_real_summary'):
        tf.summary.image('predict_real', tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope('predict_fake_summary'):
        tf.summary.image('predict_fake', tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar('discriminator_loss', model.discrim_loss)
    tf.summary.scalar('generator_loss_GAN', model.gen_loss_GAN)
    tf.summary.scalar('generator_loss_L1', model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/values', var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.name_scope('parameter_count'):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)
    sv = tf.train.Supervisor(logdir=output_path, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        parameter_count = sess.run(parameter_count)
        print('parameter_count:', parameter_count)

        max_steps = samples.steps_per_epoch * max_epochs
        print('max_steps:', max_steps)

        start = time.time()

        for step in range(max_steps):
            fetches = {
                'train': model.train,
                'global_step': sv.global_step,
                'discrim_loss': model.discrim_loss,
                'gen_loss_GAN': model.gen_loss_GAN,
                'gen_loss_L1': model.gen_loss_L1
            }
            results = sess.run(fetches)

            # global_step will have the correct step count if we resume from a checkpoint
            train_epoch = math.ceil(results['global_step'] / samples.steps_per_epoch)
            train_step = (results['global_step'] - 1) % samples.steps_per_epoch + 1
            rate = (step + 1) * batch_size / (time.time() - start)
            remaining = (max_steps - step) * batch_size / rate

            print('epoch %d step %d / %d image/sec %0.1f remaining %dm' % (train_epoch, train_step, samples.steps_per_epoch, rate, remaining / 60))
            print('discrim_loss:', results['discrim_loss'])
            print('gen_loss_GAN:', results['gen_loss_GAN'])
            print('gen_loss_L1:', results['gen_loss_L1'])

            if sv.should_stop():
                break

        print('saving model')
        saver.save(sess, os.path.join(output_path, 'model'), global_step=sv.global_step)


if __name__ == '__main__':
    output_path = '/files/_pix2pix-trainer'
    EPS = 1e-12
    image_size = 256
    Samples = collections.namedtuple('Samples', 'paths, inputs, targets, steps_per_epoch')
    Model = collections.namedtuple('Model', 'outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train')

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name')
    parser.add_argument('--imgs-path')
    parser.add_argument('--output-path', default=output_path)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--l1-weight', type=float, default=100.0, help='weight on L1 term for generator gradient')
    parser.add_argument('--gan-weight', type=float, default=1.0, help='weight on GAN term for generator gradient')
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print('{}={}'.format(k, v))

    job_name = args.job_name
    imgs_path = args.imgs_path
    output_path = args.output_path
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    ngf = args.ngf
    ndf = args.ndf
    lr = args.lr
    beta1 = args.beta1
    l1_weight = args.l1_weight
    gan_weight = args.gan_weight

    # create job output directory
    output_path = '{}/{}'.format(output_path, job_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train()
