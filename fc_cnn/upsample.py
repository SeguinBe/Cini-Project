import tensorflow as tf
import numpy as np

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    
    Args:
        :param (int): The factor to upsample the image

    Returns:
        :return Returns the size of the kernel

    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    
    Args:
        :param (int): The size of the kernel

    Returns:
        :return Returns the kernel
        
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    
    Args:
        :param (int): The factor to upsample the image
        :param (int): The number of classes on the previous layer

    Returns:
        :return Returns a matrix with weights
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights


def upsample_layer(inputs, channels_in, channels_out, factor, name="upsample"):
    """
        Create a TF layer wit the given parameters for upsampling
        
        Args:
            :param (tensor): The  previous layer input
            :param (int): Number of channels (classes) on the previous layer
            :param (int): Number of channels (classes) on the output
            :param (string): Layer name for code organization

        Returns:
            :return Returns an upsamled tensor
    """

    with tf.name_scope(name):
        up_filter = tf.constant(bilinear_upsample_weights(factor, channels_in))
        out_shape = tf.stack(
            [tf.shape(inputs)[0], inputs.get_shape()[1] * factor, inputs.get_shape()[2] * factor, channels_out])
        up_sample = tf.nn.conv2d_transpose(inputs, up_filter,
                                           output_shape=out_shape,
                                           strides=[1, factor, factor, 1])
        up_sample.set_shape([None, inputs.get_shape()[1] * factor, inputs.get_shape()[2] * factor, channels_out])
        ##
        return up_sample