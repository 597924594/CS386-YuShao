"""Code for training CycleGAN."""
from datetime import datetime
import json
import numpy as np
import os
import random
from scipy.misc import imsave
from keras.models import model_from_yaml
import sys
import gc

import click
import tensorflow as tf
import csv
from . import cyclegan_datasets
from . import losses, model
from . import data_loader
import math

##########
slim = tf.contrib.slim
GENRAL_NUM = 1
CRITIC_NUM = 1


class CycleGAN:
    """The CycleGAN module."""

    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, network_version,
                 dataset_name, checkpoint_dir, do_flipping, skip):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 1
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._network_version = network_version
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip

        self.fake_images_A = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_DEPTH, model.IMG_CHANNELS)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_DEPTH, model.IMG_CHANNELS)
        )

    def model_setup(self):
        """
        This function sets up the model to train.

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator
        of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding
        self.fake_A/self.fake_B to corresponding generator.
        This is use to calculate cyclic loss
        """

        self.g_A_input = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_HEIGHT,
                model.IMG_WIDTH,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="g_A_input")
        self.g_B_input = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_HEIGHT,
                model.IMG_WIDTH,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="g_B_input")
        self.d_A_input = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_HEIGHT,
                model.IMG_WIDTH,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="d_A_input")
        self.d_B_input = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_HEIGHT,
                model.IMG_WIDTH,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="d_B_input")

        self.cycle_A_loss_before = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_HEIGHT,
                model.IMG_WIDTH,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="cylcle_A_loss_before")
        self.cycle_B_loss_before = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_HEIGHT,
                model.IMG_WIDTH,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="cylcle_B_loss_before")

        self.d_loss_A_real = tf.placeholder(
            tf.float32,
            name="d_loss_A_real")
        self.d_loss_B_real = tf.placeholder(
            tf.float32,
            name="d_loss_B_real")
        '''self.cycle_loss_after = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_DEPTH,
                model.IMG_CHANNELS
            ], name="cylcle_loss_after")'''

        self.global_step = slim.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'd_A_input': self.d_A_input,
            'd_B_input': self.d_B_input,
            'g_A_input': self.g_A_input,
            'g_B_input': self.g_B_input,
        }
        output = model.get_models(inputs, network=self._network_version, skip=self._skip)
        self.fake_A = output['g_B_output']
        self.fake_B = output['g_A_output']
        self.cycle_A = output['g_B_output']
        self.cycle_B = output['g_A_output']

        self.fake_A_is_real = output['d_A_output']
        self.fake_B_is_real = output['d_B_output']
        self.real_A_is_real = output['d_A_output']
        self.real_B_is_real = output['d_B_output']

        self.isgan_A = output['isgan_A']
        self.isgan_B = output['isgan_B']

        self.fake_A_oc1 = output['g_B_output_oc1']
        self.fake_B_oc1 = output['g_A_output_oc1']

        self.fake_A_temp = np.zeros((1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, model.IMG_CHANNELS))
        self.fake_B_temp = np.zeros((1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, model.IMG_CHANNELS))
        self.cycle_A_temp = np.zeros((1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, model.IMG_CHANNELS))
        self.cycle_B_temp = np.zeros((1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, model.IMG_CHANNELS))
        self.inputs = {'photo': np.zeros((1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, model.IMG_CHANNELS)),
        'sketch': np.zeros((1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, model.IMG_CHANNELS))}

        self.real_A_is_real_temp = np.zeros((1))
        self.real_B_is_real_temp = np.zeros((1))

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Various trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        """
        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=self.cycle_A_loss_before, generated_images=self.cycle_B
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=self.cycle_B_loss_before, generated_images=self.cycle_A
            )

        synthesis_loss_B = \
            21 * losses.cycle_consistency_loss(
                real_images=self.g_A_input, generated_images=self.fake_A
            )
        synthesis_loss_A = \
            21 * losses.cycle_consistency_loss(
                real_images=self.g_B_input, generated_images=self.fake_B
            )

        lsgan_loss_b = losses.lsgan_loss_generator(self.isgan_B) * 5
        lsgan_loss_a = losses.lsgan_loss_generator(self.isgan_A) * 5

        oc1_loss = 1 * losses.cycle_consistency_loss(self.fake_A_oc1, self.fake_B_oc1)
        g_loss_A_cycle = cycle_consistency_loss_a
        g_loss_A_isgan = lsgan_loss_a
        '''# while training g_loss_A
        feed_dict = {self.cycle_A_loss_before : inputs['photo'],
                        self.cycle_B_loss_before: inputs['sketch'],
                        self.g_A_input: self.fake_B_temp,
                        self.g_B_input: self.fake_A_temp,
                        self.d_B_input : self.fake_B_temp,
                        self.d_A_input : self.fake_A_temp,}'''
        g_loss_B_cycle = cycle_consistency_loss_b
        g_loss_B_isgan = lsgan_loss_b

        d_loss_A_fake = tf.reduce_mean(tf.squared_difference(self.fake_A_is_real, 0))
        d_loss_B_fake = tf.reduce_mean(tf.squared_difference(self.fake_B_is_real, 0))
        d_loss_A_real = tf.reduce_mean(tf.squared_difference(self.real_A_is_real, 1))
        d_loss_B_real = tf.reduce_mean(tf.squared_difference(self.real_B_is_real, 1))
        # optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.model_vars = tf.trainable_variables()
        self.all_varsa = tf.global_variables()
        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        all_vars = [var for var in self.all_varsa]
        file = open('vars.txt', 'w')
        for var in d_A_vars: file.write(var.name+'\n')
        for var in d_B_vars: file.write(var.name+'\n')
        for var in g_A_vars: file.write(var.name+'\n')
        for var in g_B_vars: file.write(var.name+'\n')
        for var in all_vars: file.write(var.name+'\n')

        self.oc1_loss_trainer = optimizer.minimize(oc1_loss, var_list=g_A_vars+g_B_vars)

        self.d_A_trainer_real = optimizer.minimize(d_loss_A_real, var_list=d_A_vars)
        self.d_A_trainer_fake = optimizer.minimize(d_loss_A_fake, var_list=d_A_vars)
        self.d_B_trainer_real = optimizer.minimize(d_loss_B_real, var_list=d_B_vars)
        self.d_B_trainer_fake = optimizer.minimize(d_loss_B_fake, var_list=d_B_vars)

        self.g_A_trainer_syn = optimizer.minimize(synthesis_loss_A, var_list=g_A_vars)
        self.g_B_trainer_syn = optimizer.minimize(synthesis_loss_B, var_list=g_B_vars)
        self.g_A_trainer_cycle = optimizer.minimize(g_loss_A_cycle, var_list=g_A_vars)
        self.g_B_trainer_cycle = optimizer.minimize(g_loss_B_cycle, var_list=g_B_vars)
        self.g_A_trainer_isgan = optimizer.minimize(g_loss_A_isgan, var_list=g_A_vars + g_B_vars)# remain to be seen
        self.g_B_trainer_isgan = optimizer.minimize(g_loss_B_isgan, var_list=g_A_vars + g_B_vars)
##########
        # self.clip_d_A_op=[var.assign(tf.clip_by_value(
        #     var,CLIP[0],CLIP[1]))for var in d_A_vars]
        # self.clip_d_B_op=[var.assign(tf.clip_by_value(
        #     var,CLIP[0],CLIP[1]))for var in d_B_vars]
##########
        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.oc1_loss_summ = tf.summary.scalar("oc1_loss", oc1_loss)
        self.g_A_cycleloss_summ = tf.summary.scalar("g_A_loss_cycle", g_loss_A_cycle)
        self.g_B_cycleloss_summ = tf.summary.scalar("g_B_loss_cycle", g_loss_B_cycle)
        self.g_A_syn_summ = tf.summary.scalar('g_A_loss_syn', synthesis_loss_A)
        self.g_B_syn_summ = tf.summary.scalar('g_B_loss_syn', synthesis_loss_B)
        self.g_A_isganloss_summ = tf.summary.scalar('g_A_loss_isgan', g_loss_A_isgan)
        self.g_B_isganloss_summ = tf.summary.scalar('g_B_loss_isgan', g_loss_B_isgan)
        self.d_A_real_summ = tf.summary.scalar("d_A_loss_real", d_loss_A_real)
        self.d_B_real_summ = tf.summary.scalar("d_B_loss_real", d_loss_B_real)
        self.d_A_fake_summ = tf.summary.scalar("d_A_loss_fake", d_loss_A_fake)
        self.d_B_fake_summ = tf.summary.scalar("d_B_loss_fake", d_loss_B_fake)        

        #
        self.g_A_lsgan = tf.summary.scalar("g_A_lsgan", lsgan_loss_b)
        self.g_B_lsgan = tf.summary.scalar("g_B_lsgan", lsgan_loss_a)
        #

    def save_images(self, sess, epoch, batch_num):
        """
        Saves input and output images.

        :param sess: The session.
        :param epoch: Currnt epoch.
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['inputA_', 'inputB_', 'fakeA_',
                 'fakeB_', 'cycA_', 'cycB_']

        with open(os.path.join(
                self._output_dir, 'epoch_' + str(epoch) + '.html'
        ), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = self.inputs

                tensors = [inputs['photo'], inputs['sketch'],
                           self.fake_B_temp, self.fake_A_temp, self.cycle_A_temp, self.cycle_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".npy"
                    to_save = tensor[0]*255
                    to_save[to_save < 0] = 0
                    to_save[to_save > 255] = 255
                    np.save(os.path.join(self._images_dir, image_name),
                            (to_save.astype(np.uint8))
                            )
                    v_html.write(
                        "<img src=\"" +
                        os.path.join('imgs', image_name) + "\">"
                    )
                v_html.write("<br>")

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated image to corresponding
        pool of images.

        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder
        # self.inputs = data_loader.load_data(
        #     self._dataset_name, self._size_before_crop,
        #     True, self._do_flipping))

        # Build the network
        self.model_setup()
        print('model setup')
        # Loss function calculations
        self.compute_losses()
        print('compute loss')

        '''
        for tensor in tf.get_default_graph().as_graph_def().node:
            print tensor.name
            '''

        print self.g_A_input.name
        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()
        # tf.get_default_graph().finalize()

        max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # KTF.set_session(sess)
            sess.run(init)
            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                train_dataLoader = data_loader.data_loader(self._dataset_name, True)
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(
                    self._output_dir, "cyclegan"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if epoch < 100:
                    curr_lr_g = self._base_lr
                    curr_lr_d = 0.00005
                else:
                    curr_lr_g = self._base_lr - \
                        self._base_lr * (epoch - 100) / 100
                    curr_lr_d = 0.00005 - \
                        0.00005 * (epoch - 100) / 100

                for i in range(0, max_images):
                    print("Processing batch {}/{}".format(i, max_images))

                    self.save_images(sess, epoch, i)
                    inputs = train_dataLoader.getBatch(i)
                    self.inputs = inputs

                    # if i == 0:

                    # for j in range(5):
                    # Optimizing the G_A network
                    for _ in range(GENRAL_NUM):
                        _, _, _, self.fake_B_temp, self.fake_A_temp, g_A_loss_str, g_B_loss_str = sess.run([
                            self.g_A_trainer_syn,
                            self.g_B_trainer_syn,
                            self.oc1_loss_trainer,
                            self.fake_B,
                            self.fake_A,
                            self.g_A_syn_summ,
                            self.g_B_syn_summ],
                            feed_dict={
                                self.learning_rate: curr_lr_g,
                                self.g_A_input: inputs['photo'],
                                self.g_B_input: inputs['sketch'],
                            })
                        writer.add_summary(g_A_loss_str, epoch * max_images + i)
                        writer.add_summary(g_B_loss_str, epoch * max_images + i)

                        # self.store_good_fakes(epoch, i)

                        # cycleloss for G
                        _, _, self.cycle_A_temp, self.cycle_B_temp, g_A_loss_str, g_B_loss_str = sess.run([
                            self.g_A_trainer_cycle,
                            self.g_B_trainer_cycle,
                            self.cycle_A,
                            self.cycle_B,
                            self.g_A_cycleloss_summ,
                            self.g_B_cycleloss_summ
                            ],
                            feed_dict={
                                self.learning_rate: curr_lr_g,
                                self.cycle_A_loss_before: inputs['sketch'],
                                self.cycle_B_loss_before: inputs['photo'],
                                self.g_A_input: self.fake_A_temp,
                                self.g_B_input: self.fake_B_temp,
                                self.d_B_input: self.fake_B_temp,
                                self.d_A_input: self.fake_A_temp
                            })
                        writer.add_summary(g_A_loss_str, epoch * max_images + i)
                        writer.add_summary(g_B_loss_str, epoch * max_images + i)

                        # isgan for G(and for D as well)
                        _, _, isgan_a_str, isgan_b_str = sess.run([
                            self.g_A_trainer_isgan,
                            self.g_B_trainer_isgan,
                            self.g_A_isganloss_summ,
                            self.g_B_isganloss_summ
                            ],
                            feed_dict={
                                self.learning_rate: curr_lr_g,
                                self.g_A_input: inputs['photo'],
                                self.g_B_input: inputs['sketch']
                            })

                        writer.add_summary(isgan_a_str, epoch * max_images + i)
                        writer.add_summary(isgan_b_str, epoch * max_images + i)

                        if i < 25 or i % 500 == 0:
                            critic_num = 5 * CRITIC_NUM
                        else:
                            critic_num = CRITIC_NUM
                    for _ in range(critic_num):
                        # loss
                        _, _, d_A_loss_str, d_B_loss_str = sess.run([
                            self.d_A_trainer_real,
                            self.d_B_trainer_real,
                            self.d_A_real_summ,
                            self.d_B_real_summ],
                            feed_dict={
                            self.d_loss_A_real: self.real_A_is_real_temp,
                            self.d_loss_B_real: self.real_B_is_real_temp,
                            self.d_A_input: inputs['photo'],
                            self.d_B_input: inputs['sketch'],
                            self.learning_rate: curr_lr_d,
                            })
                        writer.add_summary(d_A_loss_str, epoch * max_images + i)
                        writer.add_summary(d_B_loss_str, epoch * max_images + i)

                        _, _, d_A_loss_str, d_B_loss_str = sess.run([
                            self.d_A_trainer_fake,
                            self.d_B_trainer_fake,
                            self.d_A_fake_summ,
                            self.d_B_fake_summ],
                            feed_dict={
                            self.d_loss_A_real: self.real_A_is_real_temp,
                            self.d_loss_B_real: self.real_B_is_real_temp,
                            self.d_A_input: self.fake_A_temp,
                            self.d_B_input: self.fake_B_temp,
                            self.learning_rate: curr_lr_d,
                            })
                        writer.add_summary(d_A_loss_str, epoch * max_images + i)
                        writer.add_summary(d_B_loss_str, epoch * max_images + i)

                            # Optimizing the D_A network
                        
                    writer.flush()
                    self.num_fake_inputs += 1

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)

    def test(self):
        """Test Function."""
        print("Testing the results")
        max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]
        # self.inputs = data_loader.load_data(
        #     self._dataset_name, self._size_before_crop,
        #     False, self._do_flipping)
        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        test_dataloader = data_loader.data_loader(self._dataset_name, True)
        for i in range(0, max_images):
            print("Processing batch {}/{}".format(i, max_images))

            inputs = test_dataloader.getBatch(i)
            self.inputs = inputs

            with tf.Session() as sess:
                sess.run(init)

                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                # self._num_imgs_to_save = cyclegan_datasets.DATASET_TO_SIZES[
                # self._dataset_name]
                self._num_imgs_to_save = 1
                self.save_images(sess, i)

                coord.request_stop()
                coord.join(threads)


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=True,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default=None,
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='train',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='',
              help='The name of the train/test split.')
@click.option('--skip',
              type=click.BOOL,
              default=False,
              help='Whether to add skip connection between input and output.')


def main(to_train, log_dir, config_filename, checkpoint_dir, skip):
    """

    :param to_train: Specify whether it is training or testing. 1: training; 2:
     resuming from latest checkpoint; 0: testing.
    :param log_dir: The root dir to save checkpoints and imgs. The actual dir
    is the root dir appended by the folder with the name timestamp.
    :param config_filename: The configuration file.
    :param checkpoint_dir: The directory that saves the latest checkpoint. It
    only takes effect when to_train == 2.
    :param skip: A boolean indicating whether to add skip connection between
    input and output.
    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    network_version = str(config['network_version'])
    dataset_name = str(config['dataset_name'])
    do_flipping = bool(config['do_flipping'])

    cyclegan_model = CycleGAN(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step, network_version,
                              dataset_name, checkpoint_dir, do_flipping, skip)

    if to_train > 0:
        cyclegan_model.train()
    else:
        cyclegan_model.test()


if __name__ == '__main__':
    main()
