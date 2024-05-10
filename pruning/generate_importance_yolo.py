import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from argparse import ArgumentParser
from tqdm import tqdm

import sys
sys.path.insert(0, '/root/codes/PTMM')
from utils import *
from models import yolo, yolo_loss
from dataset import load_fddb_dataset, FDDBGenerator

def calculate_cdf(input_data, max_bound, step=1e-4):
    x = np.arange(0, max_bound, step)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = np.sum(input_data < x[i]) / input_data.size
    return x, y

if __name__ == '__main__':
    model_path = '../weights/pretrain/yolo.h5'
    save_path = '../importance/yolo'
    
    os.makedirs(save_path, exist_ok=True)            

    config_device()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

    inputs = Input((112, 112, 3))
    outputs = yolo(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights(model_path, by_name=True)
        
    val_data_file = '/root/data/FDDB/VocFormat/val.txt'
    X_val, Y_val = load_fddb_dataset(val_data_file)
    val_generator = FDDBGenerator(X_val, Y_val, batch_size=32, image_shape=(112, 112))
    
    # calculate importance
    loss_fn = yolo_loss
    
    for step, (x_batch_val, y_batch_val) in tqdm(enumerate(val_generator)):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch_val, training=True)
            loss_value = loss_fn(y_batch_val, y_pred)
        grads = tape.gradient(loss_value, model.trainable_weights)

        if step == 0:
            gradients = grads[:]
        else:
            for i in range(len(grads)):
                gradients[i] += grads[i]

        if step == len(val_generator):
            break

    for i in range(len(gradients)):
        gradients[i] /= len(val_generator)

    weights_importance = []
    for w, g in zip(model.trainable_weights, gradients):
        w = w.numpy()
        g = g.numpy()
        t = (w + (np.random.random(w.shape) * 1e-20 - 5e-21)) * (g + (np.random.random(g.shape) * 1e-20 - 5e-21))
        weights_importance.append(t)
    
    # save importance
    importance = {'filter': {}, 'filterlet': {}}
    for t, w in zip(weights_importance, model.trainable_weights):
        if ('conv' in w.name or 'shortcut' in w.name) and 'kernel' in w.name:
            layer_name = w.name.split('/')[0]
            
            filter_importance = np.absolute(t).sum(axis=(0, 1, 2))
            filterlet_importance = np.absolute(t).sum(axis=2)
            
            importance['filter'][layer_name] = filter_importance
            importance['filterlet'][layer_name] = filterlet_importance
            np.save(os.path.join(save_path, '%s_filter.npy' % layer_name), filter_importance)
            np.save(os.path.join(save_path, '%s_filterlet.npy' % layer_name), filterlet_importance)
    
    # display cdf of importance
    max_bound = 0.05
    step = 1e-5
    x_lim = (0, 0.05)
    col = 3
    row = np.ceil(len(importance['filter']) / col)

    plt.figure(figsize=(14, 12))
    for idx, (layer_name, layer_importance) in enumerate(importance['filter'].items()):
        plt.subplot(row, col, idx + 1)
    #     plt.hist(layer_importance.flatten(), bins=50, density=True)
        x, cdf = calculate_cdf(layer_importance, max_bound=max_bound, step=step)
        plt.plot(x, cdf,'-',color = 'r',label='filter')
        x, cdf = calculate_cdf(importance['filterlet'][layer_name], max_bound=max_bound, step=step)
        plt.plot(x, cdf,'-',color = 'b',label='filterlet')
        plt.ylim((0, 1.1))
        plt.xlim(x_lim)
        plt.legend(loc ='lower right')
        plt.title(layer_name)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cdf.png'))


