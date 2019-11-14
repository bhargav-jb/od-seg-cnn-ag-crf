import tensorflow as tf
from tensorflow.keras import layers, models

def attn_gate(x, gate, fs, fc, act, df='channels_last'):
  """
  Our proposed attention gate for optic disc segmentation

  Args:
    x: Input tensor of shape [B, H, W, C] or [B, C, H, W]
    gate: The gating tensor of the same format as `x`
    fs: Number of filters in the spatial attention branch
    fc: Number of filters in the channel attention branch
    df: Data format of `x` and `g`, `channels_last` for [B, H, W, C]
        and `channels_first` for [B, C, W, C]

  Returns:
    att_out: The attented output after both the spatial and 
        channel based attention. Data format as specified 
        by `df`, and has the same shape as `x`.
  """
  def _spacial_att(x, g, f):
    x_shape = x.shape[1:3]
    g = tf.image.resize(g, x_shape)
    x_conv = tf.keras.layers.Conv2D(f, 1, data_format=df)(x)
    g_conv = tf.keras.layers.Conv2D(f, 1, data_format=df)(g)
    a = act()(x_conv + g_conv)
    alpha = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', data_format=df)(a)
    att_out = alpha * x
    return att_out

  def _channel_att(x, g, f):
    if df == 'channels_last':
      idx = -1
      expand_axis = [1, 2]
    else:
      idx = 1
      expand_axis = [2, 3]
    x_channels = x.shape[idx]

    x_m = tf.reduce_mean(x, [1, 2])
    g_m = tf.reduce_mean(g, [1, 2])

    x_d = tf.keras.layers.Dense(f)(x_m)
    g_d = tf.keras.layers.Dense(f)(g_m)
    a = act()(x_d + g_d)
    alpha = tf.keras.layers.Dense(x_channels, activation='softmax')(a)
    alpha = tf.expand_dims(alpha, expand_axis[0])
    alpha = tf.expand_dims(alpha, expand_axis[1])
    att_out = alpha * x
    return att_out

  return (_spacial_att(x, gate, fs) + _channel_att(x, gate, fc)) / 2.0
