import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv3d(inputconv, o_d=64, f=7, s=1, stddev=0.02,
                   padding="VALID", name="conv3d", do_norm=True, do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):
        # print inputconv.get_shape()
        conv = tf.contrib.layers.conv3d(
            inputconv, o_d, [f, f, 3], [s, s, 1], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float16
            ),
            biases_initializer=tf.constant_initializer(0.0, dtype=tf.float16)
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv3d(inputconv, o_d=64, f=7, s=1,
                     stddev=0.02, padding="VALID", name="deconv3d",
                     do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv3d_transpose(
            inputconv, o_d, [f, f, 3],
            [s, s, 1], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float16
                ),
            biases_initializer=tf.constant_initializer(0.0, dtype=tf.float16)
        )

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv
