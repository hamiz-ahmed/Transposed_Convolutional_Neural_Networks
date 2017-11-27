"""
Copyright 2017, University of Freiburg.
Muhammad Hamiz Ahmed <hamizahmed93@gmail.com>

This is the code for Deep Learning Lab Exercise 3
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def computed_loss_validation(sess, validation_xs):
    global output
    calculated_loss_validation = sess.run(loss, feed_dict={xs: validation_xs})
    return calculated_loss_validation


def plot_images(images, cls_true, cls_pred=None):
    img_shape = (28, 28)
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def calculate_output_shape(in_layer, n_kernel):
    in_shape = in_layer.get_shape()  # assumes in_shape[0] = None or batch_size
    out_shape = [s for s in in_shape]  # copy
    out_shape[-1] = n_kernel # always true
    out_shape[1] = in_shape[1]*2
    out_shape[2] = in_shape[2]*2
    return out_shape


def create_convolution_layer(input_image, num_input_channels, num_filters,
                             filter_size=3,
                             use_pooling=True):

    # creates a convolution layer by default

    weight_shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight_variable(shape=weight_shape)
    biases = bias_variable([num_filters])

    layer = conv2d(input_image, weights)
    layer = tf.nn.relu(layer + biases)

    if use_pooling:
        layer = max_pool_2x2(layer)
    return layer


def create_transposed_convolution(input_image, num_input_channels):
    return conv2d_transpose(input_image, num_input_channels)


def conv2d_transpose(x, num_input_channels):
    return tf.layers.conv2d_transpose(inputs=x, filters=num_input_channels,
                                      kernel_size=2,
                                      strides=2,
                                      padding='same')


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def show_images(sess):
    y_pre = sess.run(output, feed_dict={xs: data.validation.images[0:9]})
    y_pre = np.reshape(y_pre, [9, 28, 28])
    plot_images(images=images, cls_true=cls_true)
    plot_images(images=y_pre, cls_true=cls_true)


def perform_operations():
    learning_rates = [0.1, 0.01, 0.001]

    for rate in learning_rates:
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        print("Learning rate: ", rate)
        y_plot_validation = []
        x_plot_validation = []

        for i in range(1000):
            batch_xs, batch_ys = data.train.next_batch(64)
            sess.run(train_step, feed_dict={xs: batch_xs, learning_rate: rate})

            if i % 50 == 0:
                print("Step: ", i)
                loss = computed_loss_validation(sess, data.validation.images)
                print("Validation loss: ", loss)
                x_plot_validation.append(i)
                y_plot_validation.append(loss)

        plt.plot(x_plot_validation, y_plot_validation, label=rate)
        # show_images(sess)
        sess.close()

    # for plotting graph
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load mnist data
    data = input_data.read_data_sets("data/", one_hot=True)
    data_class = np.array([label.argmax() for label in data.validation.labels])
    images = data.validation.images[0:9]
    cls_true = data_class[0:9]

    # # define placeholder for inputs to network
    learning_rate = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # keep_prob = tf.placeholder(tf.float32)

    # conv pooling layer 1
    layer1 = create_convolution_layer(x_image, 1, 8, use_pooling=True)

    # conv pooling layer 2
    layer2 = create_convolution_layer(layer1, 8, 4, use_pooling=True)

    # conv layer 3
    layer3 = create_convolution_layer(layer2, 4, 2, use_pooling=False)

    # trans conv layer 4
    layer4 = create_transposed_convolution(layer3, 2) #4

    layer55 = create_convolution_layer(layer4, 2, 4, use_pooling=False)

    # trans conv layer 5
    layer5 = create_transposed_convolution(layer55, 4) # 8 returned

    layer6 = create_convolution_layer(layer5, 4, 8, use_pooling=False)

    # output conv
    output = create_convolution_layer(layer6, 8, 1, use_pooling=False)

    output = tf.reshape(output, [tf.shape(output)[0], 784])

    loss = tf.reduce_mean(tf.square(output - xs))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    perform_operations()
