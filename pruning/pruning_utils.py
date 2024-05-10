import numpy as np
import copy
import graph


def generate_mask(model, filter_importance, filterlet_importance, alpha, beta):
    GRAPH = graph.get_graph()
    filter_mask = {}
    filterlet_mask = {}
    mask = {}
    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            idx = node['conv_id']
            name = node['name']
            f = filter_importance[idx]
            flet = copy.deepcopy(filterlet_importance[idx])
            weights = model.get_layer(name).get_weights()[0]
            kernel_size, _, in_channel, out_channel = weights.shape
            f_p = np.percentile(f, beta[idx] * 100)
            filter_mask[name] = np.zeros(weights.shape, dtype=bool)
            filter_mask[name][:, :, :, f <= f_p] = 1
            flet[:, :, f <= f_p] = np.max(flet)
            flet_p = np.percentile(flet, alpha[idx] * 100)
            filterlet_mask[name] = np.zeros(
                (in_channel, kernel_size, kernel_size, out_channel),
                dtype=bool)
            filterlet_mask[name][:, flet <= flet_p] = 1
            filterlet_mask[name] = np.transpose(filterlet_mask[name],
                                                (1, 2, 0, 3))
            mask[name] = filter_mask[name] | filterlet_mask[name]
    return mask, filter_mask, filterlet_mask


def generate_mask_with_admm(model, filter_importance, filterlet_importance, alpha, beta):
    GRAPH = graph.get_graph()
    filter_mask = {}
    filterlet_mask = {}
    mask = {}
    for node in GRAPH:
        if node['type'] == 'conv':
            idx = node['conv_id']
            name = node['name']
            weights = model.get_layer(name).get_weights()[0]
            f = np.sum(np.abs(weights), axis=(0, 1, 2))
            flet = np.sum(np.abs(weights), axis=2)
            kernel_size, _, in_channel, out_channel = weights.shape
            f_p = np.percentile(f, beta[idx] * 100)
            filter_mask[name] = np.zeros(weights.shape, dtype=bool)
            filter_mask[name][:, :, :, f <= f_p] = 1
            flet[:, :, f <= f_p] = np.max(flet)
            flet_p = np.percentile(flet, alpha[idx] * 100)
            filterlet_mask[name] = np.zeros(
                (in_channel, kernel_size, kernel_size, out_channel),
                dtype=bool)
            filterlet_mask[name][:, flet <= flet_p] = 1
            filterlet_mask[name] = np.transpose(filterlet_mask[name],
                                                (1, 2, 0, 3))
            mask[name] = filter_mask[name] | filterlet_mask[name]
    return mask, filter_mask, filterlet_mask


def set_weights_to_zero(model, mask):
    GRAPH = graph.get_graph()
    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            name = node['name']
            layer = model.get_layer(name)
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            weights[mask[name]] = 0
            layer.set_weights([weights, biases])


def generate_mask_from_weights(model):
    GRAPH = graph.get_graph()
    mask = {}
    for node in GRAPH:
        if node['type'] == 'conv':
            name = node['name']
            layer = model.get_layer(name)
            weights = layer.get_weights()[0]
            kernel_size, _, in_channel, out_channel = weights.shape
            m = np.zeros(weights.shape, dtype=bool)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    for l in range(out_channel):
                        if np.all(weights[i, j, :, l] == 0):
                            m[i, j, :, l] = 1
            mask[name] = m
    return mask


# def make_dict(model, alpha, beta, filter_importance, filterlet_importance):
#     GRAPH = graph.get_graph()
#     Z_dict = {}
#     U_dict = {}
#     for node in GRAPH:
#         if node['type'] == 'conv':
#             layer = model.get_layer(node['name'])
#             Z_dict[node['name']] = layer.get_weights()[0]
#             U_dict[node['name']] = np.zeros_like(layer.get_weights()[0])
#     Z_dict = projection(Z_dict, alpha, beta, filter_importance, filterlet_importance)
#     return Z_dict, U_dict


# def projection(Z_dict, alpha, beta, filter_importance, filterlet_importance):
#     for i, key in enumerate(Z_dict.keys()):
#         f = filter_importance[i]
#         flet = copy.deepcopy(filterlet_importance[i])
#         kernel_size, _, in_channel, out_channel = Z_dict[key].shape
#         f_p = np.percentile(f, beta[i] * 100)
#         filter_mask = np.zeros(Z_dict[key].shape, dtype=bool)
#         filter_mask[:, :, :, f <= f_p] = 1
#         flet[:, :, f <= f_p] = np.max(flet)
#         flet_p = np.percentile(flet, alpha[i] * 100)
#         filterlet_mask = np.zeros((in_channel, kernel_size, kernel_size, out_channel), dtype=bool)
#         filterlet_mask[:, flet <= flet_p] = 1
#         filterlet_mask = np.transpose(filterlet_mask, (1, 2, 0, 3))
#         mask = filter_mask | filterlet_mask
#         Z_dict[key] = Z_dict[key] * (1 - mask)
#     return Z_dict


def make_dict(model, alpha, beta):
    GRAPH = graph.get_graph()
    Z_dict = {}
    U_dict = {}
    for node in GRAPH:
        if node['type'] == 'conv':
            layer = model.get_layer(node['name'])
            Z_dict[node['name']] = layer.get_weights()[0]
            U_dict[node['name']] = np.zeros_like(layer.get_weights()[0])
    Z_dict = projection(Z_dict, alpha, beta)
    return Z_dict, U_dict



def projection(Z_dict, alpha, beta):
    
    for i, key in enumerate(Z_dict.keys()):
        f = np.sum(np.abs(Z_dict[key]), axis=(0, 1, 2))
        flet = np.sum(np.abs(Z_dict[key]), axis=2)
        kernel_size, _, in_channel, out_channel = Z_dict[key].shape
        f_p = np.percentile(f, beta[i] * 100)
        filter_mask = np.zeros(Z_dict[key].shape, dtype=bool)
        filter_mask[:, :, :, f <= f_p] = 1
        flet[:, :, f <= f_p] = np.max(flet)
        flet_p = np.percentile(flet, alpha[i] * 100)
        filterlet_mask = np.zeros((in_channel, kernel_size, kernel_size, out_channel), dtype=bool)
        filterlet_mask[:, flet <= flet_p] = 1
        filterlet_mask = np.transpose(filterlet_mask, (1, 2, 0, 3))
        mask = filter_mask | filterlet_mask
        Z_dict[key] = Z_dict[key] * (1 - mask)
    return Z_dict