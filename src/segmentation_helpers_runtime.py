import cv2
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import layers
import time
from scipy.interpolate import splev, splprep




@tf.function
def deeplumen_segmentation(image, model, original_image=None):

    OFFSET = tf.constant([60.3486, 60.3486, 60.3486], dtype=tf.float32)
    # add batch dimension
    tensor_reshaped = tf.expand_dims(image, axis=0)
    tensor_reshaped = tf.cast(tensor_reshaped, tf.float32) - OFFSET

    # forward pass
    logits = model(tensor_reshaped, training=False)
    logits = tf.image.resize(logits, (224, 224))

    probs = tf.nn.softmax(logits, axis=-1)
    pred = tf.argmax(logits, axis=-1, output_type=tf.dtypes.int32)

    conf_class2 = None
    conf_colormap = None
    overlay = None

    # use static shape check
    if probs.shape[-1] > 2:
        conf_class2 = probs[0, ..., 2]



    return pred, conf_class2



def post_process_deeplumen(raw_data, conf_class2, conf_threshold, hybrid_seg=None):
    """
    Optimized version of the original function — identical outputs, faster execution.
    """

    # ---------- CLASS SPLITTING ---------- #
    mask_1 = np.uint8(raw_data == 1) * 255
    mask_2 = np.uint8(raw_data == 2) * 255

    # ---------- MASK 1 PROCESSING ---------- #
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_1, connectivity=8)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_1 = np.uint8(labels == largest_label) * 255

        # contour extraction
        contours, _ = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            # largest contour by length
            contour = max(contours, key=len)
            contour = contour.squeeze()  # (N, 2)

            # spline fit — subsample if very dense for speed
            if contour.ndim == 2 and len(contour) > 20:
                x, y = contour[:, 0], contour[:, 1]
                tck, _ = splprep([x, y], s=500.0, per=True)
                new_points = np.linspace(0, 1, len(x))
                spline = np.array(splev(new_points, tck)).T
                spline_pixels = np.round(spline).astype(np.int32).reshape(-1, 1, 2)

                mask_1[:] = 0  # reuse array
                cv2.fillPoly(mask_1, [spline_pixels], 255)
            else:
                spline_pixels = contour.reshape(-1, 1, 2)
        else:
            spline_pixels = np.empty((0, 1, 2), np.int32)
    else:
        spline_pixels = np.empty((0, 1, 2), np.int32)

    # ---------- MASK 2 CONFIDENCE FILTERING ---------- #
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_2, connectivity=8)
    largest_two_masks = []

    if num_labels > 1:
        comp_areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        comp_labels = np.arange(1, num_labels)

        # vectorized max confidence per label (slightly faster than loop)
        valid_labels = []
        valid_areas = []

        for lbl, area in zip(comp_labels, comp_areas):
            region_mask = (labels == lbl)
            if not np.any(region_mask):
                continue
            comp_conf = conf_class2[region_mask].max()
            if comp_conf >= conf_threshold:
                valid_labels.append(lbl)
                valid_areas.append(area)

        if valid_labels:
            top_idx = np.argsort(valid_areas)[-2:][::-1]
            mask_2[:] = 0  # reuse array
            for idx in top_idx:
                lbl = valid_labels[idx]
                comp_mask = np.uint8(labels == lbl) * 255
                largest_two_masks.append(comp_mask)
                mask_2 |= comp_mask
        else:
            mask_2[:] = 0
    else:
        mask_2[:] = 0

    # ---------- REMOVE OVERLAPS ----------
    overlap = (mask_1 > 0) & (mask_2 > 0)
    if np.any(overlap):
        mask_1[overlap] = 0

        

    return mask_1, mask_2, largest_two_masks, spline_pixels






class BlurPool(layers.Layer):
    def __init__(self, kernel_size, filters, strides=2):
        super(BlurPool, self).__init__()
        self.strides = (1, strides, strides, 1)
        self.kernel_size = kernel_size
        self.filters = filters
        self.padding = ((int(1.*(kernel_size-1)/2), int(tf.math.ceil(1.*(kernel_size-1)/2))), (int(1.*(kernel_size-1)/2), int(tf.math.ceil(1.*(kernel_size-1)/2))))
        if self.kernel_size == 1:
            a = tf.constant([1.,])
        elif self.kernel_size == 2:
            a = tf.constant([1., 1.])
        elif self.kernel_size == 3:
            a = tf.constant([1., 2., 1.])
        elif self.kernel_size == 4:
            a = tf.constant([1., 3., 3., 1.])
        elif self.kernel_size == 5:
            a = tf.constant([1., 4., 6. ,4., 1.])
        elif self.kernel_size == 6:
            a = tf.constant([1., 5., 10. ,10., 5., 1.])
        elif self.kernel_size == 7:
            a = tf.constant([1. ,6., 15., 20., 15., 6., 1.])

        a = a[:, None]*a[None, :]
        a = a/tf.reduce_sum(a)
        a = tf.tile(a[:, :, None, None], (1, 1, filters, 1))
        self.filter = tf.constant(a, dtype=tf.float32)



    def compute_output_shape(self, input_shape):
        height = math.ceil(input_shape[1] / self.strides[0]) if input_shape[1] is not None else None
        width  = math.ceil(input_shape[2] / self.strides[1]) if input_shape[2] is not None else None
        channels = input_shape[3]
        return (input_shape[0], height, width, channels)



    @tf.function
    def call(self, x):

        x = tf.nn.depthwise_conv2d(x, self.filter, strides=self.strides, padding='SAME')

        return x

    def get_config(self):
        config = super(BlurPool, self).get_config()
        config.update({'kernel_size': self.kernel_size, 'filters': self.filters, 'strides': self.strides})
        return config



regularizer = tf.keras.regularizers.l2(1e-3)
init        = tf.keras.initializers.HeNormal()
act         = tf.keras.layers.ReLU()
act.__name__ = "relu"

# --- Utility: BlurPool wrapper must exist in your codebase ---
# expected signature: BlurPool(kernel_size=3, filters=<int>)
# It should apply a low-pass (blur) then a stride-2 downsample.
# ----------------------------------------------------------------

def ConvBN(filters, k, dilation=1, name=None):
    def f(x):
        x = layers.Conv2D(filters, k, padding="same", dilation_rate=dilation,
                          use_bias=False, kernel_initializer=init,
                          kernel_regularizer=regularizer, name=None if name is None else name+"_conv")(x)
        x = layers.BatchNormalization(momentum=0.9, name=None if name is None else name+"_bn")(x)
        x = layers.Activation(act, name=None if name is None else name+"_act")(x)
        return x
    return f

def MLDRBlock(filters, dilations=(1,2,4), name=None):
    """
    Multi-Level Dilated Residual block:
      parallel 3x3 atrous convs on the SAME input → concat → 1x1 fuse → residual add → ReLU
    """
    def f(x):
        inp = x
        branches = []
        for d in dilations:
            branches.append(ConvBN(filters, 3, dilation=d, name=None if name is None else f"{name}_d{d}")(x))
        x = layers.Concatenate(name=None if name is None else f"{name}_concat")(branches)
        # fuse to 'filters' channels
        x = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                          kernel_initializer=init, kernel_regularizer=regularizer,
                          name=None if name is None else f"{name}_fuse_conv")(x)
        x = layers.BatchNormalization(momentum=0.9, name=None if name is None else f"{name}_fuse_bn")(x)

        # project residual if needed
        if inp.shape[-1] != filters:
            inp = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                kernel_initializer=init, kernel_regularizer=regularizer,
                                name=None if name is None else f"{name}_proj_conv")(inp)
            inp = layers.BatchNormalization(momentum=0.9, name=None if name is None else f"{name}_proj_bn")(inp)
        x = layers.Add(name=None if name is None else f"{name}_add")([x, inp])
        x = layers.Activation(act, name=None if name is None else f"{name}_out")(x)
        return x
    return f

def DownBlock(filters, name=None):
    """
    Residual downsample using BlurPool on both paths.
    """
    def f(x):
        inp = x
        # main branch: conv → BN → act → conv → BN → BlurPool
        x = ConvBN(filters, 3, name=None if name is None else f"{name}_c1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_initializer=init, kernel_regularizer=regularizer,
                          name=None if name is None else f"{name}_c2_conv")(x)
        x = layers.BatchNormalization(momentum=0.9, name=None if name is None else f"{name}_c2_bn")(x)
        x = BlurPool(kernel_size=3, filters=filters)(x)

        # skip: 1x1 then BlurPool to match spatial/channel dims
        skip = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                             kernel_initializer=init, kernel_regularizer=regularizer,
                             name=None if name is None else f"{name}_skip_conv")(inp)
        skip = layers.BatchNormalization(momentum=0.9, name=None if name is None else f"{name}_skip_bn")(skip)
        skip = BlurPool(kernel_size=3, filters=filters)(skip)

        x = layers.Add(name=None if name is None else f"{name}_add")([x, skip])
        x = layers.Activation(act, name=None if name is None else f"{name}_out")(x)
        return x
    return f

def CBAM(x, reduction=0.25, spatial_kernel=7, name=None):
    """Keras-compatible CBAM block (channel + spatial attention)."""
    ch = x.shape[-1]

    # ----- Channel Attention -----
    gap = layers.GlobalAveragePooling2D(keepdims=True, name=None if name is None else f"{name}_gap")(x)
    gmp = layers.GlobalMaxPooling2D(keepdims=True, name=None if name is None else f"{name}_gmp")(x)

    shared_dense1 = layers.Dense(int(ch * reduction), activation="relu", use_bias=False,
                                 name=None if name is None else f"{name}_fc1")
    shared_dense2 = layers.Dense(ch, activation="sigmoid", use_bias=False,
                                 name=None if name is None else f"{name}_fc2")

    ca_avg = shared_dense2(shared_dense1(gap))
    ca_max = shared_dense2(shared_dense1(gmp))
    ca = layers.Add(name=None if name is None else f"{name}_ca_add")([ca_avg, ca_max])
    x = layers.Multiply(name=None if name is None else f"{name}_ch_scale")([x, ca])

    # ----- Spatial Attention -----
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                             name=None if name is None else f"{name}_sa_avg")(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                             name=None if name is None else f"{name}_sa_max")(x)
    concat = layers.Concatenate(axis=-1, name=None if name is None else f"{name}_sa_concat")([avg_pool, max_pool])

    sa = layers.Conv2D(1, spatial_kernel, padding="same", activation="sigmoid", use_bias=False,
                       name=None if name is None else f"{name}_sa_conv")(concat)
    x = layers.Multiply(name=None if name is None else f"{name}_sa_scale")([x, sa])

    return x


def build_mldr_drn(input_shape=(None, None, 3), num_classes=3,
                   base=16, blocks_per_stage=(1,1,2,2,2), dilations=(1,2,4),
                   dropout=0.2, upsample_stride=8, return_pyramid=False, name="MLDR_DRN"):
    """
    Multi-Level DRN with MLDR blocks and BlurPool downsamples.
    Encoder returns pyramid C2..C5; decoder upsamples from 1/8 stride to full res.

    blocks_per_stage: how many MLDR blocks per stage AFTER the initial DownBlock.
    """
    inp = layers.Input(shape=input_shape, name="img")

    # Stem (no downsample yet)
    x = layers.Conv2D(base, 7, padding="same", use_bias=False,
                      kernel_initializer=init, kernel_regularizer=regularizer, name="stem_conv")(inp)
    x = layers.BatchNormalization(momentum=0.9, name="stem_bn")(x)
    x = layers.Activation(act, name="stem_act")(x)

    # Stage 1 (stride 2)
    x = DownBlock(base, name="down1")(x)
    for i in range(blocks_per_stage[0]):
        x = MLDRBlock(base, dilations, name=f"s1_b{i+1}")(x)
    c2 = x  # ~1/2

    # Stage 2 (stride 4)
    x = DownBlock(base*2, name="down2")(x)
    for i in range(blocks_per_stage[1]):
        x = MLDRBlock(base*2, dilations, name=f"s2_b{i+1}")(x)
    c3 = x  # ~1/4

    # Stage 3 (stride 8)
    x = DownBlock(base*4, name="down3")(x)
    for i in range(blocks_per_stage[2]):
        x = MLDRBlock(base*4, dilations, name=f"s3_b{i+1}")(x)
    c4 = x  # ~1/8

    # Stage 4 (stride 8, no further downsample; larger receptive field via dilations)
    for i in range(blocks_per_stage[3]):
        x = MLDRBlock(base*4, dilations=(2,4,8), name=f"s4_b{i+1}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    c5 = x  # context @ 1/8

    # Optional: deeper context stage with even larger dilations
    for i in range(blocks_per_stage[4]):
        x = MLDRBlock(base*4, dilations=(1,2,4), name=f"s5_b{i+1}")(x)
    c6 = x  # still 1/8

    if return_pyramid:
        return tf.keras.Model(inp, [c2, c3, c4, c5, c6], name=name)

    # --- Lightweight decoder with nonlinear skip refinement ---
    # bring c3 and c2 to c4 resolution for rich fusion
    def up_to(x, target, name):
      """Resize x to exactly match the spatial size of target."""
      def resize_like(tensors):
          x, target = tensors
          target_shape = tf.shape(target)
          return tf.image.resize(x, [target_shape[1], target_shape[2]], method="bilinear")
      return layers.Lambda(resize_like, name=name)([x, target])


    # CBAM only
    c3r = CBAM(c3)
    c2r = CBAM(c2)

    # fuse c6 + c4
    f = layers.Concatenate(name="fuse_c6_c4")([c6, c4])
    f = ConvBN(base*4, 3, name="dec_fuse1a")(f)
    f = ConvBN(base*4, 3, name="dec_fuse1b")(f)

    # up to c3
    f = up_to(f, c3r, "up_to_c3")
    f = layers.Concatenate(name="fuse_c3")([f, c3r])
    f = ConvBN(base*2, 3, name="dec_fuse2a")(f)
    f = ConvBN(base*2, 3, name="dec_fuse2b")(f)

    # up to c2
    f = up_to(f, c2r, "up_to_c2")
    f = layers.Concatenate(name="fuse_c2")([f, c2r])
    f = ConvBN(base, 3, name="dec_fuse3a")(f)
    f = ConvBN(base, 3, name="dec_fuse3b")(f)

    # logits at 1/2 stride; upsample to full
    logits_low = layers.Conv2D(num_classes, 1, padding="same", kernel_initializer=init,
                               kernel_regularizer=regularizer, name="logits")(f)

    # final upsample back to input size (we downsampled 2× then 2× then 2× → stride=8)
    out = layers.UpSampling2D(size=(upsample_stride, upsample_stride),
                              interpolation="bilinear", name="upsample_to_full")(logits_low)

    return tf.keras.Model(inp, out, name=name)







