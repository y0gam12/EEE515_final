from __future__ import division
from scipy.io import loadmat
from scipy.io import savemat
import glob
import os, time, scipy.io
#import tensorflow as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim
import numpy as np
from tflearn.layers.conv import global_avg_pool
from networkv2 import network
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import cv2


_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

#val_dir = './Validation/ValidationNoisyBlocksRaw.mat'
checkpoint_dir = 'C:/Users/yogam\Documents/School/EEE515/checkpoint/SIDD_Pyramid/model.ckpt'
result_dir = 'C:/Users/yogam/Documents/School/EEE515/yoga_res/'
img_dir = 'C:/Users/yogam/Documents/School/EEE515/finalproject_testimgs/*'

#mat = loadmat(val_dir)
# print(mat.keys)
#val_img = mat['ValidationNoisyBlocksRaw'] #(40, 32, 256, 256)
# val_img = mat['ValidationNoisyBlocksRaw']
# val_img = np.expand_dims(val_img.reshape([1280, 256, 256]), axis=3)
# val_img = val_img.reshape([1280, 256, 256])
img_list = glob.glob(img_dir)
in_img_list = []
for img_path in img_list:
    img = cv2.imread(img_path)
    fix_size = cv2.resize(img, (256,256))
    gray_img = cv2.cvtColor(fix_size, cv2.COLOR_BGR2GRAY)
    in_img_list.append(gray_img)


ps = 256


ouput_blocks = [None] * 40 * 32

sess = tf.compat.v1.Session()#sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 1])
# gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = network(in_image)

saver = tf.train.Saver(max_to_keep=15)
sess.run(tf.global_variables_initializer())

print('loaded ' + checkpoint_dir)
saver.restore(sess, checkpoint_dir)


if not os.path.isdir(result_dir):
    print('-----------------------------no existing path')
    os.makedirs(result_dir)

for i in range(len(in_img_list)):

    each_block = in_img_list[i] #(256, 256)
    each_block = np.expand_dims(np.expand_dims(each_block, axis=0), axis=3)
    

    st = time.time()
    output = sess.run(out_image, feed_dict={in_image: each_block})
    output = np.minimum(np.maximum(output, 0), 1)

    
    t_cost = time.time() - st
    ouput_blocks[i] = output
    print(ouput_blocks[i].shape)
    #scipy.misc.toimage(each_block[0,:,:,0] * 255, high=255, low=0, cmin=0, cmax=255).save(
    #            result_dir + '%04d_test_in.jpg' % (i))
    #scipy.misc.toimage(output[0,:,:,0] * 255, high=255, low=0, cmin=0, cmax=255).save(
    #            result_dir + '%04d_test_out.jpg' % (i))
    in_img = toimage(each_block[0,:,:,0] * 255, high=255, low=0, cmin=0, cmax=255, pal=None, mode=None, channel_axis=None)
    in_path = result_dir + '%04d_test_in.jpg' % (i)
    in_img.save(in_path)

    out_img = toimage(output[0,:,:,0] * 255, high=255, low=0, cmin=0, cmax=255, pal=None, mode=None, channel_axis=None)
    out_path = result_dir + '%04d_test_out.jpg' % (i)
    out_img.save(out_path)

    print('cleaning block %4d' % i)
    print('time_cost:', t_cost)
#out_mat = np.squeeze(ouput_blocks)
#out_mat = out_mat.reshape([40, 32, 256, 256])

#savemat(result_dir + 'ValidationCleanBlocksRaw.mat', {'results': out_mat})
