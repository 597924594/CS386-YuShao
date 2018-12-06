"""Code for constructing the model and get the outputs from the model."""
import numpy as np
import tensorflow as tf
import scipy.ndimage
from . import layers
import pdb

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

IMG_DEPTH = 3

# The number of color channels per image.
IMG_CHANNELS = 1

POOL_SIZE = 50
# ngf = 32
# ndf = 64
ngf = 128
ndf = 128
# 20180512-194528 128 128


def get_models(inputs, network="tensorflow", skip=False):
    d_A_input = inputs['d_A_input']
    d_B_input = inputs['d_B_input']
    g_A_input = inputs['g_A_input']
    g_B_input = inputs['g_B_input']

    with tf.variable_scope("Model") as scope:
        if network == "tensorflow":
            current_discriminator = discriminator_tf
            current_generator = build_generator_new
        d_A_output = current_discriminator(d_A_input, 'd_A')
        d_B_output = current_discriminator(d_B_input, 'd_B')
        g_A_output, g_A_output_oc1 = build_generator_new4(g_A_input, 'g_A', skip=skip)
        g_B_output, g_B_output_oc1 = build_generator_new4(g_B_input, 'g_B', skip=skip)

        scope.reuse_variables()
        isgan_A = current_discriminator(g_A_output, 'd_B')
        isgan_B = current_discriminator(g_B_output, 'd_A')
    return {'d_A_output': d_A_output,
            'd_B_output': d_B_output,
            'g_A_output': g_A_output,
            'g_B_output': g_B_output,
            'isgan_B': isgan_B,
            'isgan_A': isgan_A,
            'g_A_output_oc1': g_A_output_oc1,
            'g_B_output_oc1': g_B_output_oc1,
            }


# def discriminator_tf(inputdisc, name="discriminator"):
#     with tf.variable_scope(name):
#         f = 4

#         o_c1 = layers.general_conv3d(inputdisc, ndf, f, 2,
#                                      0.02, "SAME", "c1", do_norm=False,
#                                      relufactor=0.2)
#         o_c2 = layers.general_conv3d(o_c1, ndf * 2, f, 2,
#                                      0.02, "SAME", "c2", relufactor=0.2)
#         o_c3 = layers.general_conv3d(o_c2, ndf * 4, f, 2,
#                                      0.02, "SAME", "c3", relufactor=0.2)
#         o_c4 = layers.general_conv3d(o_c3, ndf * 8, f, 1,
#                                      0.02, "SAME", "c4", relufactor=0.2)
#         o_c4 = layers.general_conv3d(
#             o_c6, 1, f, 1, 0.02,
#             "SAME", "c5", do_norm=False, do_relu=False
#         )

#         return o_c5

# generator from High-Quality Facial Photo-Sketch Synthesis Using Multi-Adersarial Networks
def build_generator_new(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f1 = 7
        f2 = 3
        # o_c1 = layers.general_conv3d(inputgen, num of filters , size of filters , stride , stddev = 0.02, padding = "SAME", name = "c1")
        o_c1 = layers.general_conv3d(
            inputgen, 64, f1, 1, 0.02, "SAME", "c1")  # C7S1-64
        o_c2 = layers.general_conv3d(
            o_c1, 128, f2, 2, 0.02, "SAME", "c2")  # C3-128
        o_c3 = layers.general_conv3d(
            o_c2, 256, f2, 2, 0.02, "SAME", "c3")  # C3-256
        # ##################Residual Block#################################
        # Residual Block 1
        o_c4 = layers.general_conv3d(
            o_c3, 256, f2, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f2, 1, 0.02, "SAME", "c5")
        # Residual Block 2
        o_c6 = layers.general_conv3d(
            o_c5, 256, f2, 1, 0.02, "SAME", "c6")
        o_c7 = layers.general_conv3d(
            o_c6, 256, f2, 1, 0.02, "SAME", "c7")
        # Residual Block 3
        o_c8 = layers.general_conv3d(
            o_c7, 256, f2, 1, 0.02, "SAME", "c8")
        o_c9 = layers.general_conv3d(
            o_c8, 256, f2, 1, 0.02, "SAME", "c9")
        # Residual Block 4
        o_c10 = layers.general_conv3d(
            o_c9, 256, f2, 1, 0.02, "SAME", "c10")
        o_c11 = layers.general_conv3d(
            o_c10, 256, f2, 1, 0.02, "SAME", "c11")
        # Residual Block 5
        o_c12 = layers.general_conv3d(
            o_c11, 256, f2, 1, 0.02, "SAME", "c12")
        o_c13 = layers.general_conv3d(
            o_c12, 256, f2, 1, 0.02, "SAME", "c13")
        # Residual Block 6
        o_c14 = layers.general_conv3d(
            o_c13, 256, f2, 1, 0.02, "SAME", "c14")
        o_c15 = layers.general_conv3d(
            o_c14, 256, f2, 1, 0.02, "SAME", "c15")
        # Residual Block 7
        o_c16 = layers.general_conv3d(
            o_c15, 256, f2, 1, 0.02, "SAME", "c16")
        o_c17 = layers.general_conv3d(
            o_c16, 256, f2, 1, 0.02, "SAME", "c17")
        # Residual Block 8
        o_c18 = layers.general_conv3d(
            o_c17, 256, f2, 1, 0.02, "SAME", "c18")
        o_c19 = layers.general_conv3d(
            o_c18, 256, f2, 1, 0.02, "SAME", "c19")
        # Residual Block 9
        o_c20 = layers.general_conv3d(
            o_c19, 256, f2, 1, 0.02, "SAME", "c20")
        o_c21 = layers.general_conv3d(
            o_c20, 256, f2, 1, 0.02, "SAME", "c21")
        # ##################Residual Block#################################
        o_c22 = layers.general_deconv3d(
            o_c21, 64, f2, 2, 0.02, "SAME", "c22")  # TC64
        o_c23 = layers.general_deconv3d(
            o_c22, 32, f2, 2, 0.02, "SAME", "c23")  # TC32
        out_gen = layers.general_deconv3d(
            o_c23, 1, f1, 1, 0.02, "SAME", "c24")  # C7S1-3

        print ('*****')
        print (out_gen.get_shape())

        print (out_gen.name)
        return out_gen, o_c1


def build_generator_new2(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f1 = 7
        f2 = 3
        # o_c1 = layers.general_conv3d(inputgen, num of filters , size of filters , stride , stddev = 0.02, padding = "SAME", name = "c1")
        o_c1 = layers.general_conv3d(
            inputgen, 64, f1, 1, 0.02, "SAME", "c1")  # C7S1-64
        o_c2 = layers.general_conv3d(
            o_c1, 128, f2, 2, 0.02, "SAME", "c2")  # C3-128
        o_c3 = layers.general_conv3d(
            o_c2, 256, f2, 2, 0.02, "SAME", "c3")  # C3-256
        # Residual Block 1
        o_c4 = layers.general_conv3d(
            o_c3, 256, f2, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f2, 1, 0.02, "SAME", "c5")

        identity_1 = tf.concat([o_c3, o_c5], 4)

        # Residual Block 2
        o_c6 = layers.general_conv3d(
            identity_1, 256, f2, 1, 0.02, "SAME", "c6")
        o_c7 = layers.general_conv3d(
            o_c6, 256, f2, 1, 0.02, "SAME", "c7")

        identity_2 = tf.concat([identity_1, o_c7], 4)

        # Residual Block 3
        o_c8 = layers.general_conv3d(
            identity_2, 256, f2, 1, 0.02, "SAME", "c8")
        o_c9 = layers.general_conv3d(
            o_c8, 256, f2, 1, 0.02, "SAME", "c9")

        identity_3 = tf.concat([identity_2, o_c9], 4)

        # Residual Block 4
        o_c10 = layers.general_conv3d(
            identity_3, 256, f2, 1, 0.02, "SAME", "c10")
        o_c11 = layers.general_conv3d(
            o_c10, 256, f2, 1, 0.02, "SAME", "c11")

        identity_4 = tf.concat([identity_3, o_c11], 4)

        # Residual Block 5
        o_c12 = layers.general_conv3d(
            identity_4, 256, f2, 1, 0.02, "SAME", "c12")
        o_c13 = layers.general_conv3d(
            o_c12, 256, f2, 1, 0.02, "SAME", "c13")

        identity_5 = tf.concat([identity_4, o_c13], 4)

        # Residual Block 6
        o_c14 = layers.general_conv3d(
            identity_5, 256, f2, 1, 0.02, "SAME", "c14")
        o_c15 = layers.general_conv3d(
            o_c14, 256, f2, 1, 0.02, "SAME", "c15")

        identity_6 = tf.concat([identity_5, o_c15], 4)

        # Residual Block 7
        o_c16 = layers.general_conv3d(
            identity_6, 256, f2, 1, 0.02, "SAME", "c16")
        o_c17 = layers.general_conv3d(
            o_c16, 256, f2, 1, 0.02, "SAME", "c17")

        identity_7 = tf.concat([identity_6, o_c17], 4)

        # Residual Block 8
        o_c18 = layers.general_conv3d(
            identity_7, 256, f2, 1, 0.02, "SAME", "c18")
        o_c19 = layers.general_conv3d(
            o_c18, 256, f2, 1, 0.02, "SAME", "c19")

        identity_8 = tf.concat([identity_7, o_c19], 4)

        # Residual Block 9
        o_c20 = layers.general_conv3d(
            identity_8, 256, f2, 1, 0.02, "SAME", "c20")
        o_c21 = layers.general_conv3d(
            o_c20, 256, f2, 1, 0.02, "SAME", "c21")

        identity_9 = tf.concat([identity_8, o_c21], 4)

        #############################################################
        o_c22 = layers.general_deconv3d(
            identity_9, 64, f2, 2, 0.02, "SAME", "c22")  # TC64
        o_c23 = layers.general_deconv3d(
            o_c22, 32, f2, 2, 0.02, "SAME", "c23")  # TC32
        out_gen = layers.general_deconv3d(
            o_c23, 1, f1, 1, 0.02, "SAME", "c24")  # C7S1-3

        print ('*****')
        print (out_gen.get_shape())

        print (out_gen.name)
        return out_gen


def build_generator_new3(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f1 = 7
        f2 = 3
        # o_c1 = layers.general_conv3d(inputgen, num of filters , size of filters , stride , stddev = 0.02, padding = "SAME", name = "c1")
        o_c1 = layers.general_conv3d(
            inputgen, 64, f1, 1, 0.02, "SAME", "c1")  # C7S1-64

        o_c2 = layers.general_conv3d(
            o_c1, 128, f2, 2, 0.02, "SAME", "c2")  # C3-128

        o_c3 = layers.general_conv3d(
            o_c2, 256, f2, 2, 0.02, "SAME", "c3")  # C3-256

        # ##################Residual Block#################################
        # Residual Block 1
        o_c4 = layers.general_conv3d(
            o_c3, 256, f2, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f2, 1, 0.02, "SAME", "c5")

        # Residual Block 2
        o_c6 = layers.general_conv3d(
            o_c5, 256, f2, 1, 0.02, "SAME", "c6")
        o_c7 = layers.general_conv3d(
            o_c6, 256, f2, 1, 0.02, "SAME", "c7")

        # Residual Block 3
        o_c8 = layers.general_conv3d(
            o_c7, 256, f2, 1, 0.02, "SAME", "c8")
        o_c9 = layers.general_conv3d(
            o_c8, 256, f2, 1, 0.02, "SAME", "c9")

        # Residual Block 4
        o_c10 = layers.general_conv3d(
            o_c9, 256, f2, 1, 0.02, "SAME", "c10")
        o_c11 = layers.general_conv3d(
            o_c10, 256, f2, 1, 0.02, "SAME", "c11")

        # Residual Block 5
        o_c12 = layers.general_conv3d(
            o_c11, 256, f2, 1, 0.02, "SAME", "c12")
        o_c13 = layers.general_conv3d(
            o_c12, 256, f2, 1, 0.02, "SAME", "c13")

        # Residual Block 6
        o_c14 = layers.general_conv3d(
            o_c13, 256, f2, 1, 0.02, "SAME", "c14")
        o_c15 = layers.general_conv3d(
            o_c14, 256, f2, 1, 0.02, "SAME", "c15")

        identity_1 = tf.concat([o_c11, o_c15], 4)
        # Residual Block 7
        o_c16 = layers.general_conv3d(
            identity_1, 256, f2, 1, 0.02, "SAME", "c16")
        o_c17 = layers.general_conv3d(
            o_c16, 256, f2, 1, 0.02, "SAME", "c17")

        identity_2 = tf.concat([o_c9, o_c17], 4)
        # Residual Block 8
        o_c18 = layers.general_conv3d(
            identity_2, 256, f2, 1, 0.02, "SAME", "c18")
        o_c19 = layers.general_conv3d(
            o_c18, 256, f2, 1, 0.02, "SAME", "c19")

        identity_3 = tf.concat([o_c7, o_c19], 4)
        # Residual Block 9
        o_c20 = layers.general_conv3d(
            identity_3, 256, f2, 1, 0.02, "SAME", "c20")
        o_c21 = layers.general_conv3d(
            o_c20, 256, f2, 1, 0.02, "SAME", "c21")

        identity_4 = tf.concat([o_c5, o_c21], 4)
        # ##################Residual Block#################################
        o_c22 = layers.general_deconv3d(
            identity_4, 64, f2, 2, 0.02, "SAME", "c22")  # TC64

        identity_5 = tf.concat([o_c2, o_c22], 4)
        o_c23 = layers.general_deconv3d(
            identity_5, 32, f2, 2, 0.02, "SAME", "c23")  # TC32

        identity_6 = tf.concat([o_c1, o_c23], 4)

        out_gen = layers.general_deconv3d(
            identity_6, 1, f1, 1, 0.02, "SAME", "c24")  # C7S1-3

        print ('*****')
        print (out_gen.get_shape())

        print (out_gen.name)
        return out_gen, o_c1


def build_generator_new4(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f1 = 7
        f2 = 3
        # o_c1 = layers.general_conv3d(inputgen, num of filters , size of filters , stride , stddev = 0.02, padding = "SAME", name = "c1")
        o_c1 = layers.general_conv3d(
            inputgen, 64, f1, 1, 0.02, "SAME", "c1")  # C7S1-64
        o_c2 = layers.general_conv3d(
            o_c1, 128, f2, 2, 0.02, "SAME", "c2")  # C3-128
        o_c3 = layers.general_conv3d(
            o_c2, 256, f2, 2, 0.02, "SAME", "c3")  # C3-256
        # ##################Residual Block#################################
        # Residual Block 1
        o_c4 = layers.general_conv3d(
            o_c3, 256, f2, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f2, 1, 0.02, "SAME", "c5")
        # Residual Block 2
        o_c6 = layers.general_conv3d(
            o_c5, 256, f2, 1, 0.02, "SAME", "c6")
        o_c7 = layers.general_conv3d(
            o_c6, 256, f2, 1, 0.02, "SAME", "c7")

        identity_1 = tf.concat([o_c5, o_c7], 4)
        # Residual Block 3
        o_c8 = layers.general_conv3d(
            identity_1, 256, f2, 1, 0.02, "SAME", "c8")
        o_c9 = layers.general_conv3d(
            o_c8, 256, f2, 1, 0.02, "SAME", "c9")
        identity_2 = tf.concat([identity_1, o_c9], 4)

        # Residual Block 4
        o_c10 = layers.general_conv3d(
            identity_2, 256, f2, 1, 0.02, "SAME", "c10")
        o_c11 = layers.general_conv3d(
            o_c10, 256, f2, 1, 0.02, "SAME", "c11")
        identity_3 = tf.concat([identity_2, o_c11], 4)

        # Residual Block 5
        o_c12 = layers.general_conv3d(
            identity_3, 256, f2, 1, 0.02, "SAME", "c12")
        o_c13 = layers.general_conv3d(
            o_c12, 256, f2, 1, 0.02, "SAME", "c13")
        identity_4 = tf.concat([identity_3, o_c13], 4)

        # Residual Block 6
        o_c14 = layers.general_conv3d(
            identity_4, 256, f2, 1, 0.02, "SAME", "c14")
        o_c15 = layers.general_conv3d(
            o_c14, 256, f2, 1, 0.02, "SAME", "c15")
        identity_5 = tf.concat([identity_4, o_c15], 4)

        # Residual Block 7
        o_c16 = layers.general_conv3d(
            identity_5, 256, f2, 1, 0.02, "SAME", "c16")
        o_c17 = layers.general_conv3d(
            o_c16, 256, f2, 1, 0.02, "SAME", "c17")
        identity_6 = tf.concat([identity_5, o_c17], 4)

        # Residual Block 8
        o_c18 = layers.general_conv3d(
            identity_6, 256, f2, 1, 0.02, "SAME", "c18")
        o_c19 = layers.general_conv3d(
            o_c18, 256, f2, 1, 0.02, "SAME", "c19")
        identity_7 = tf.concat([identity_6, o_c19], 4)

        # Residual Block 9
        o_c20 = layers.general_conv3d(
            identity_7, 256, f2, 1, 0.02, "SAME", "c20")
        o_c21 = layers.general_conv3d(
            o_c20, 256, f2, 1, 0.02, "SAME", "c21")
        identity_8 = tf.concat([identity_7, o_c21], 4)

        # ##################Residual Block#################################
        o_c22 = layers.general_deconv3d(
            identity_8, 64, f2, 2, 0.02, "SAME", "c22")  # TC64

        identity_9 = tf.concat([o_c2, o_c22], 4)

        o_c23 = layers.general_deconv3d(
            identity_9, 32, f2, 2, 0.02, "SAME", "c23")  # TC32
        out_gen = layers.general_deconv3d(
            o_c23, 1, f1, 1, 0.02, "SAME", "c24")  # C7S1-3

        print ('*****')
        print (out_gen.get_shape())

        print (out_gen.name)
        return out_gen, o_c1


def discriminator_tf(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 3
        # o_c0 = layers.general_conv3d(inputdisc[:,17:29,17:29,17:29,:],48, f, 2,
        #                              0.02, "SAME", "c0", do_norm=False,
        #                              relufactor=0.2)
        o_c1 = layers.general_conv3d(inputdisc, ndf, f, 2,
                                     0.02, "SAME", "c1", do_norm=False,
                                     relufactor=0.2)
        o_c2 = layers.general_conv3d(o_c1, ndf * 2, f, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv3d(o_c2, ndf * 4, f, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv3d(o_c3, ndf * 8, f, 1,
                                     0.02, "SAME", "c4", relufactor=0.2)

        # print o_c0.get_shape()
        print '*************'
        # o_c6 = tf.concat([o_c0,o_c4],4)
        # o_c5 = layers.general_conv3d(
        #     o_c6, 1, f, 1, 0.02,
        #     "SAME", "c5", do_norm=False, do_relu=False
        # )
        o_c5 = layers.general_conv3d(
            o_c4, 1, f, 1, 0.02,
            "SAME", "c5", do_norm=False, do_relu=False
        )
        print o_c5.get_shape()
        return o_c5


def discriminator_tf_deeper(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 3
        # o_c0 = layers.general_conv3d(inputdisc[:,17:29,17:29,17:29,:],48, f, 2,
        #                              0.02, "SAME", "c0", do_norm=False,
        #                              relufactor=0.2)
        o_c1 = layers.general_conv3d(inputdisc, ndf, f, 1,
                                     0.02, "SAME", "c1", do_norm=False,
                                     relufactor=0.2)
        o_c2 = layers.general_conv3d(o_c1, ndf, f, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv3d(o_c2, ndf * 2, f, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv3d(o_c3, ndf * 2, f, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv3d(o_c4, ndf * 4, f, 2,
                                     0.02, "SAME", "c5", relufactor=0.2)
        o_c6 = layers.general_conv3d(o_c5, ndf * 4, f, 1,
                                     0.02, "SAME", "c6", relufactor=0.2)
        o_c7 = layers.general_conv3d(o_c6, ndf * 8, f, 1,
                                     0.02, "SAME", "c7", relufactor=0.2)
        o_c8 = layers.general_conv3d(o_c7, ndf * 8, f, 1,
                                     0.02, "SAME", "c8", relufactor=0.2)
        # print o_c0.get_shape()
        # print '*************'
        # print o_c4.get_shape()
        # o_c6 = tf.concat([o_c0,o_c4],4)
        # o_c5 = layers.general_conv3d(
        #     o_c6, 1, f, 1, 0.02,
        #     "SAME", "c5", do_norm=False, do_relu=False
        # )
        o_c9 = layers.general_conv3d(
            o_c8, 1, f, 1, 0.02,
            "SAME", "c9", do_norm=False, do_relu=False
        )

        print '*****'
        print o_c9.get_shape()
        print o_c9.name
        return o_c9


def patch_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 3

        # patch_input = tf.random_crop(inputdisc, [48, 48, 48,1])
        o_c1 = layers.general_conv3d(inputdisc, ndf, f, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv3d(o_c1, ndf * 2, f, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv3d(o_c2, ndf * 4, f, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv3d(o_c3, ndf * 8, f, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv3d(
            o_c4, 1, f, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5


def build_generator_tf_test(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 3
        ks = 4
        padding = "REFLECT"

        # pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks,ks], [0, 0]], padding)
        o_c1 = layers.general_conv3d(
            inputgen, 16, f, 2, 0.02, "SAME", "c1")  # 24
        o_c2 = layers.general_conv3d(
            o_c1, 32, f, 1, 0.02, "SAME", "c2")
        o_c3 = layers.general_conv3d(
            o_c2, 64, f, 2, 0.02, "SAME", "c3")  # 12
        o_c4 = layers.general_conv3d(
            o_c3, 128, f, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f, 2, 0.02, "SAME", "c5")  # 6
        o_c6 = layers.general_conv3d(
            o_c5, 512, f, 1, 0.02, "SAME", "c6")
        o_c7 = layers.general_conv3d(
            o_c6, 1024, f, 2, 0.02, "SAME", "c7")  # 3
        # print o_c6.get_shape()
        # print o_c7.get_shape()
        o_c8 = layers.general_conv3d(
            o_c7, 2048, f, 1, 0.02, "SAME", 'c8')
        o_c9 = layers.general_conv3d(
            o_c8, 2048, f, 1, 0.02, "SAME", 'c9')
        o_c10 = tf.concat([o_c8, o_c9], 4)
        o_c11 = layers.general_deconv3d(  # 6
            o_c10, 1024, f, 2, 0.02,
            "SAME", "c11")
        o_c12 = layers.general_conv3d(
            o_c11, 512, f, 1, 0.02, "SAME", "c12")
        o_c13 = tf.concat([o_c6, o_c12], 4)
        o_c14 = layers.general_conv3d(
            o_c13, 256, f, 1, 0.02, "SAME", "c14")
        o_c15 = layers.general_deconv3d(
            o_c14, 128, f, 2, 0.02,
            "SAME", "c15")  # 12
        o_c16 = layers.general_conv3d(
            o_c15, 64, f, 1, 0.02, "SAME", "c16")
        o_c17 = tf.concat([o_c3, o_c16], 4)
        o_c18 = layers.general_deconv3d(
            o_c17, 32, f, 2, 0.02,
            "SAME", "c18")  # 24
        o_c19 = layers.general_conv3d(
            o_c18, 16, f, 1, 0.02, "SAME", "c19")
        o_c20 = tf.concat([o_c1, o_c19], 4)
        o_c21 = layers.general_deconv3d(
            o_c20, 8, f, 2, 0.02, "SAME", "c21")  # 48
        out_gen = layers.general_conv3d(
            o_c21, 1, f, 1, 0.02, "SAME", "c22")
        print '*****'
        print out_gen.get_shape()

        return out_gen


def build_generator_tf(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 3
        ks = 3
        padding = "REFLECT"

        # pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks,ks], [0, 0]], padding)
        o_c1 = layers.general_conv3d(
            inputgen, 16, f, 2, 0.02, "SAME", "c1")
        o_c2 = layers.general_conv3d(
            o_c1, 32, f, 1, 0.02, "SAME", "c2")
        o_c3 = layers.general_conv3d(
            o_c2, 64, f, 2, 0.02, "SAME", "c3")
        o_c4 = layers.general_conv3d(
            o_c3, 128, f, 1, 0.02, "SAME", "c4")
        o_c5 = layers.general_conv3d(
            o_c4, 256, f, 2, 0.02, "SAME", "c5")
        o_c6 = layers.general_conv3d(
            o_c5, 512, f, 1, 0.02, "SAME", "c6")
        o_c7 = layers.general_conv3d(
            o_c6, 512, f, 1, 0.02, "SAME", "c7")
        # print o_c6.get_shape()

        # print o_c7.get_shape()
        o_c8 = tf.concat([o_c6, o_c7], 4)
        o_c9 = layers.general_deconv3d(
            o_c8, 256, f, 2, 0.02,
            "SAME", "c9")
        o_c10 = layers.general_conv3d(
            o_c9, 128, f, 1, 0.02, "SAME", "c10")
        o_c11 = tf.concat([o_c4, o_c10], 4)
        o_c12 = layers.general_conv3d(
            o_c11, 64, f, 1, 0.02, "SAME", "c12")
        o_c13 = layers.general_deconv3d(
            o_c12, 32, f, 2, 0.02,
            "SAME", "c13")
        o_c14 = layers.general_conv3d(
            o_c13, 16, f, 1, 0.02, "SAME", "c14")
        o_c15 = tf.concat([o_c2, o_c14], 4)
        o_c16 = layers.general_deconv3d(
            o_c15, 8, f, 2, 0.02,
            "SAME", "c16")
        out_gen = layers.general_conv3d(
            o_c16, 1, f, 1, 0.02, "SAME", "c17")
        print '*****'
        print out_gen.get_shape()

        print out_gen.name
        return out_gen
