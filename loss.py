import tensorflow as tf

def focal_loss_mask(hm_pred, hm_true, mask):
    pos_mask = tf.cast(tf.equal(hm_true, 1.), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1.), dtype=tf.float32)
    neg_weights = tf.pow(1. - hm_true, 4)

    #area_mask = tf.cast(tf.greater(mask, 0.), dtype=tf.float32)
    one = tf.ones_like(mask)
    zero = tf.zeros_like(mask)
    W = tf.where(mask >= 0.5, x=one, y=zero)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(1. - hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1. - hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(hm_pred, 2.0) * neg_weights * neg_mask * W

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    loss = tf.cond(tf.greater(num_pos, 0), lambda : (pos_loss + neg_loss) / num_pos, lambda : neg_loss)
    return loss

def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1.), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1.), dtype=tf.float32)
    neg_weights = tf.pow(1. - hm_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(1. - hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1. - hm_pred, 1e-5, 1. - 1e-5)) * tf.pow(hm_pred, 2.0) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    loss = tf.cond(tf.greater(num_pos, 0), lambda : (pos_loss + neg_loss) / num_pos, lambda : neg_loss)
    return loss

def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.batch_gather(y_pred, indices)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, c))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
    return loss

def reg_l1_loss_kpt(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.batch_gather(y_pred, indices)
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
    return loss

def giou(self, boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / tf.maximum(union_area, 1e-5)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-5)

    return giou


def dice_coefficient1(y_pred_cls, y_true_cls):
    '''
    dice loss
    :param y_true_cls: ground truth
    :param y_pred_cls: predict
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
    union = tf.reduce_sum(y_true_cls) + tf.reduce_sum(y_pred_cls) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return loss

def dice_coefficient(y_pred_cls, y_true_cls, training_mask):
    eps = 1e-5
    y_true_cls = y_true_cls * training_mask
    y_pred_cls = y_pred_cls * training_mask
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
    union = tf.reduce_sum(y_true_cls * y_true_cls) + tf.reduce_sum(y_pred_cls * y_pred_cls) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return loss

def cross_entropy_loss(y_true_cls, y_pred_cls, training_mask):
    y_true_cls = y_true_cls * training_mask
    y_pred_cls = y_pred_cls * training_mask
    loss =  tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
    loss = tf.reduce_mean(loss)
    return loss