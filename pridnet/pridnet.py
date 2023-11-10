import torch
import torch.nn as nn

# TF Tensor: [batch, width, height, in_channels]
# Torch Tensor: [batch, in_channels, iH, iW]
def lrelu(x):
    return torch.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    # Torch Filter shape(in_channels, out_channels/groups, kH, kW)
    # TF Filter shape (height, width, output_channels, in_channels)
    deconv_filter = torch.empty(in_channels, output_channels, pool_size, pool_size)
    nn.trunc_normal(deconv_filter, std=0.02)
    deconv = nn.functional.conv_transpose2d(x1, deconv_filter, x2.shape(dim=0), strides=(pool_size, pool_size)) # Stride may be wrong

    deconv_output = torch.concat((deconv, x2), dim=3)
    deconv_output.reshape(deconv_output, (None, None, None, output_channels * 2))

    return deconv_output

def unet(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu)
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu)
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv1 = nn.functional.conv2d(input, )