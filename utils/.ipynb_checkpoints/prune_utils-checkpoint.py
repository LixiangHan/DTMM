import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def get_gradients(data_generator, model, steps):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    for step, (x_batch_train, y_batch_train) in enumerate(data_generator):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, y_pred)
        step_grads = tape.gradient(loss_value, model.trainable_weights)

        if step == 0:
            grads = step_grads[:]
        else:
            for i in range(len(grads)):
                grads[i] += step_grads[i]

        if step == steps:
            break
    # for i in range(len(grads)):
    #     grads[i] += 1
    return grads


def calculate_taylor(model, gradients):
    taylor = []
    for w, g in zip(model.trainable_weights, gradients):
        w = w.numpy()
        g = g.numpy()
        t = (w + (np.random.random(w.shape) * 1e-3 - 5e-4)) * (g + (np.random.random(g.shape) * 1e-3 - 5e-4))
        # t = w * g
        taylor.append(t)
    return taylor


def calculate_score(model, taylor):
    score = {'filter': [], 'vector': []}
    for t, w in zip(taylor, model.trainable_weights):
        if ('conv' in w.name or 'shortcut' in w.name
            ) and 'kernel' in w.name:  # is kernel weights of conv layer
            k_h, k_w, in_channel, out_channel = w.shape

            filter_score = np.absolute(t.sum(axis=(0, 1, 2), keepdims=True))
            # filter_score = np.absolute(t).sum(axis=(0, 1, 2), keepdims=True)
            filter_score /= (in_channel * k_h * k_w)
            filter_score /= np.linalg.norm(filter_score)
            # filter_score /= np.max(filter_score )
            filter_score = np.repeat(filter_score, k_h, axis=0)
            filter_score = np.repeat(filter_score, k_w, axis=1)
            filter_score = np.repeat(filter_score, in_channel, axis=2)

            vector_score = np.absolute(t.sum(axis=2, keepdims=True))
            # vector_score = np.absolute(t).sum(axis=2, keepdims=True)
            vector_score /= in_channel
            vector_score /= np.linalg.norm(vector_score)
            # vector_score /= np.max(vector_score)
            vector_score = np.repeat(vector_score, in_channel, axis=2)

            score['filter'].append(filter_score)
            score['vector'].append(vector_score)
    return score


def generate_masks_resnet12(model, score, alpha, beta):
    filter_mask = {}
    vector_mask = {}
    mask = {}
    layer_name = [
        w.name.split('/')[0] for w in model.trainable_weights
        if ('conv' in w.name or 'shortcut' in w.name) and 'kernel' in w.name
    ]
    for i, (name, f,
            v) in enumerate(zip(layer_name, score['filter'], score['vector'])):
        if name == 'block_1_conv_2':
            k_h, k_w, in_channel, out_channel = f.shape

            filter_mask[name] = filter_mask['conv_0'][0, 0, 0, :].reshape(
                (1, 1, 1, filter_mask['conv_0'].shape[-1]))
            filter_mask[name] = np.repeat(filter_mask[name], k_h, axis=0)
            filter_mask[name] = np.repeat(filter_mask[name], k_w, axis=1)
            filter_mask[name] = np.repeat(filter_mask[name],
                                          in_channel,
                                          axis=2)

            v[filter_mask[name]] = v.max()
            v_percentile = np.percentile(v, beta[i] * 100)
            vector_mask[name] = (v <= v_percentile)
        elif 'shortcut' in name:
            pre_name = name.replace('shortcut', 'conv_2')
            k_h, k_w, in_channel, out_channel = f.shape
            filter_mask[name] = filter_mask[pre_name][0, 0, 0, :].reshape(
                (1, 1, 1, filter_mask[pre_name].shape[-1]))
            filter_mask[name] = np.repeat(filter_mask[name], k_h, axis=0)
            filter_mask[name] = np.repeat(filter_mask[name], k_w, axis=1)
            filter_mask[name] = np.repeat(filter_mask[name],
                                          in_channel,
                                          axis=2)

            v[filter_mask[name]] = v.max()
            v_percentile = np.percentile(v, beta[i] * 100)
            vector_mask[name] = (v <= v_percentile)

        else:
            f_percentile = np.percentile(f, alpha[i] * 100)
            filter_mask[name] = (f <= f_percentile)
            v[filter_mask[name]] = v.max()
            v_percentile = np.percentile(v, beta[i] * 100)
            vector_mask[name] = (v <= v_percentile)
        mask[name] = filter_mask[name] | vector_mask[name]

    return mask, filter_mask, vector_mask


def generate_masks(model, score, alpha, beta):
    filter_mask = {}
    vector_mask = {}
    mask = {}
    layer_name = [
        w.name.split('/')[0] for w in model.trainable_weights
        if (('conv' in w.name or 'shortcut' in w.name) and ('conv_out' not in w.name)) and 'kernel' in w.name
    ]
    for i, (name, f,
            v) in enumerate(zip(layer_name, score['filter'], score['vector'])):
        f_percentile = np.percentile(f, alpha[i] * 100)
        filter_mask[name] = (f <= f_percentile)
        v[filter_mask[name]] = v.max()
        v_percentile = np.percentile(v, beta[i] * 100)
        vector_mask[name] = (v <= v_percentile)
        mask[name] = filter_mask[name] | vector_mask[name]
    return mask, filter_mask, vector_mask


