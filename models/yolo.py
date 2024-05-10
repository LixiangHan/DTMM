import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l1_l2



def yolo(inputs):
    # block 1
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_1_conv_1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_1_conv_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(2, 2)(x)

    # block 2
    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_2_conv_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_2_conv_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(2, 2)(x)

    # block 3
    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_3_conv_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_3_conv_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(2, 2)(x)

    # block 4
    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_4_conv_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_4_conv_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_4_conv_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(2, 2)(x)

    # block 5
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_5_conv_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_5_conv_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='block_5_conv_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(2, 2)(x)

    x = layers.Conv2D(filters=7 * 7 * (1 + 5 * 2),
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer= l1_l2(1e-5, 1e-4),
                      name='conv_out')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # x = Reshape((7, 7, 11))(x)
    x = layers.Reshape((7, 7, 11))(x)
    x = layers.Activation('sigmoid')(x)
    return x


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats, image_size=448):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * image_size
    box_wh = feats[..., 2:4] * image_size

    return box_xy, box_wh

def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :1]  # ? * 7 * 7 * 20
    label_box = y_true[..., 1:5]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 5]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :1]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., 1:3]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 3:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box, image_size=112)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box, image_size=112)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box, image_size=112)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box, image_size=112)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 112)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 112)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss