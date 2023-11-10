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
    # Convs 1
    convolution2d = nn.Conv2d(input.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv1 = convolution2d(input)
    conv1 = lrelu(conv1)
    convolution2d = nn.Conv2d(conv1.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv1 = convolution2d(conv1)
    conv1 = lrelu(conv1)
    convolution2d = nn.Conv2d(conv1.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv1 = convolution2d(conv1)
    conv1 = lrelu(conv1)
    convolution2d = nn.Conv2d(conv1.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv1 = convolution2d(conv1)
    conv1 = lrelu(conv1)
    pool1 = nn.functional.max_pool2d(conv1, kernal_size = (2,2), stride = 2, padding = 0 ) #https://stackoverflow.com/questions/62166719/padding-same-conversion-to-pytorch-padding

    # Convs 2
    convolution2d = nn.Conv2d(pool1.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv2 = convolution2d(pool1)
    conv2 = lrelu(conv2)
    convolution2d = nn.Conv2d(conv2.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv2 = convolution2d(conv2)
    conv2 = lrelu(conv2)
    convolution2d = nn.Conv2d(conv2.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv2 = convolution2d(conv2)
    conv2 = lrelu(conv2)
    convolution2d = nn.Conv2d(conv2.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv2 = convolution2d(conv2)
    conv2 = lrelu(conv2)
    pool2 = nn.functional.max_pool2d(conv2, kernal_size = (2,2), stride = 2, padding = 0 )

    # Convs 3
    convolution2d = nn.Conv2d(pool2.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv3 = convolution2d(pool2)
    conv3 = lrelu(conv3)
    convolution2d = nn.Conv2d(conv3.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv3 = convolution2d(conv3)
    conv3 = lrelu(conv3)
    convolution2d = nn.Conv2d(conv3.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv3 = convolution2d(conv3)
    conv3 = lrelu(conv3)
    convolution2d = nn.Conv2d(conv3.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv3 = convolution2d(conv3)
    conv3 = lrelu(conv3)
    pool3 = nn.functional.max_pool2d(conv3, kernal_size = (2,2), stride = 2, padding = 0 )

    # Convs 4
    convolution2d = nn.Conv2d(pool3.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv4 = convolution2d(pool3)
    conv4 = lrelu(conv4)
    convolution2d = nn.Conv2d(conv4.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv4 = convolution2d(conv4)
    conv4 = lrelu(conv4)
    convolution2d = nn.Conv2d(conv4.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv4 = convolution2d(conv4)
    conv4 = lrelu(conv4)
    convolution2d = nn.Conv2d(conv4.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv4 = convolution2d(conv4)
    conv4 = lrelu(conv4)
    pool4 = nn.functional.max_pool2d(conv4, kernal_size = (2,2), stride = 2, padding = 0 )

    # Convs 5
    convolution2d = nn.Conv2d(pool4.size(dim=1), 512, (3,3), padding = 'same', dilation = 1)
    conv5 = convolution2d(pool4)
    conv5 = lrelu(conv5)
    convolution2d = nn.Conv2d(conv5.size(dim=1), 512, (3,3), padding = 'same', dilation = 1)
    conv5 = convolution2d(conv5)
    conv5 = lrelu(conv5)
    convolution2d = nn.Conv2d(conv5.size(dim=1), 512, (3,3), padding = 'same', dilation = 1)
    conv5 = convolution2d(conv5)
    conv5 = lrelu(conv5)
    convolution2d = nn.Conv2d(conv5.size(dim=1), 512, (3,3), padding = 'same', dilation = 1)
    conv5 = convolution2d(conv5)
    conv5 = lrelu(conv5)

    # Convs 6
    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    convolution2d = nn.Conv2d(up6.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv6 = convolution2d(up6)
    conv6 = lrelu(conv6)
    convolution2d = nn.Conv2d(conv6.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv6 = convolution2d(conv6)
    conv6 = lrelu(conv6)
    convolution2d = nn.Conv2d(conv6.size(dim=1), 256, (3,3), padding = 'same', dilation = 1)
    conv6 = convolution2d(conv6)
    conv6 = lrelu(conv6)

    # Convs 7
    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    convolution2d = nn.Conv2d(up7.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv7 = convolution2d(up7)
    conv7 = lrelu(conv7)
    convolution2d = nn.Conv2d(conv7.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv7 = convolution2d(conv7)
    conv7 = lrelu(conv7)
    convolution2d = nn.Conv2d(conv7.size(dim=1), 128, (3,3), padding = 'same', dilation = 1)
    conv7 = convolution2d(conv7)
    conv7 = lrelu(conv7)

    # Convs 8
    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    convolution2d = nn.Conv2d(up8.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv8 = convolution2d(up8)
    conv8 = lrelu(conv8)
    convolution2d = nn.Conv2d(conv8.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv8 = convolution2d(conv8)
    conv8 = lrelu(conv8)
    convolution2d = nn.Conv2d(conv8.size(dim=1), 64, (3,3), padding = 'same', dilation = 1)
    conv8 = convolution2d(conv8)
    conv8 = lrelu(conv8)

    # Convs 9
    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    convolution2d = nn.Conv2d(up9.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv9 = convolution2d(up9)
    conv9 = lrelu(conv9)
    convolution2d = nn.Conv2d(conv9.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv9 = convolution2d(conv9)
    conv9 = lrelu(conv9)
    convolution2d = nn.Conv2d(conv9.size(dim=1), 32, (3,3), padding = 'same', dilation = 1)
    conv9 = convolution2d(conv9)
    conv9 = lrelu(conv9)


    convolution2d = nn.Conv2d(conv9.size(dim=1), 1, (1,1), padding = 'same', dilation = 1)
    conv10 = convolution2d(conv9)
    return conv10

