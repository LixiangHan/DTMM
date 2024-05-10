import numpy as np
import graph


def calculate_model_size(alpha, beta):
    GRAPH = graph.get_graph()
    model_size = 0
    for node in GRAPH:
        layer_size = 0
        if node['from'][0] != -1:
            if node['type'] == 'conv' and node['prune'] == True:
                kernel_size, _, input_channels, output_channels = node['weights_shape']
                conv_id = node['conv_id']
                parent_node = GRAPH[node['from'][0]]
                if parent_node['type'] == 'add':
                    parent_parent_node_0 = GRAPH[parent_node['from'][0]]
                    input_channels_after_prune = int(
                        parent_parent_node_0['weights_shape'][3] * \
                        (1 - beta[parent_parent_node_0['conv_id']]))
                else:
                    parent_node = GRAPH[node['from'][0]]
                    input_channels_after_prune = int(parent_node['weights_shape'][3] * \
                                                    (1 - beta[parent_node['conv_id']]))
                output_channels_after_prune = int(output_channels *
                                                  (1 - beta[conv_id]))
                if int(alpha[conv_id] * np.prod([kernel_size, kernel_size, output_channels])) == 0 or kernel_size == 1:
                    layer_size = kernel_size**2 * output_channels_after_prune * input_channels_after_prune
                # print(output_channels_after_prune, input_channels)
                else:
                    num_filterlet_num_after_prune = np.prod([
                        kernel_size, kernel_size, output_channels_after_prune
                    ]) - int(alpha[conv_id] *
                             np.prod([kernel_size, kernel_size, output_channels]))
                    num_weights = num_filterlet_num_after_prune * input_channels_after_prune
                # TODO: calculating the size of quantization parameters would be more accurate
                    layer_size = num_weights + 2 * num_filterlet_num_after_prune + 2 * output_channels_after_prune
            elif node['type'] == 'fc':
                parent_node = GRAPH[node['from'][0]]
                parent_parent_node = GRAPH[parent_node['from'][0]]
                input_neuros, output_neuros = node['weights_shape']
                input_neuros = int(input_neuros *
                                   (1 - beta[parent_parent_node['conv_id']]))

                num_weights = input_neuros * output_neuros + output_neuros
                layer_size = num_weights
            elif node['type'] == 'add':
                pass
            elif node['type'] == 'conv' and node['prune'] == False:
                kernel_size, _, input_channels, output_channels = node['weights_shape']
                conv_id = node['conv_id']
                parent_node = GRAPH[node['from'][0]]
                if parent_node['type'] == 'add':
                    parent_parent_node_0 = GRAPH[parent_node['from'][0]]
                    input_channels_after_prune = int(
                        parent_parent_node_0['weights_shape'][3] * \
                        (1 - beta[parent_parent_node_0['conv_id']]))
                else:
                    parent_node = GRAPH[node['from'][0]]
                    input_channels_after_prune = int(parent_node['weights_shape'][3] * \
                                                    (1 - beta[parent_node['conv_id']]))
                output_channels_after_prune = output_channels
                layer_size = kernel_size**2 * output_channels_after_prune * input_channels_after_prune
            else:
                print(node['type'], 'is not supported yet!')
        else:
            conv_id = node['conv_id']
            kernel_size, _, input_channels, output_channels = node['weights_shape']
            output_channels_after_prune = int(output_channels *
                                              (1 - beta[conv_id]))

            if int(alpha[node['conv_id']] * np.prod([kernel_size, kernel_size, output_channels])) or kernel_size == 1:
                layer_size = kernel_size**2 * input_channels * output_channels_after_prune
            else:
                num_filterlet_num_after_prune = np.prod([
                    kernel_size, kernel_size, output_channels_after_prune
                ]) - int(alpha[node['conv_id']] *
                         np.prod([kernel_size, kernel_size, output_channels]))
                num_weights = num_filterlet_num_after_prune * input_channels
                layer_size = num_weights + 2 * num_filterlet_num_after_prune + 2 * output_channels_after_prune
        model_size += layer_size
    return model_size