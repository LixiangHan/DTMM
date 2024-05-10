import sys
import graph

sys.path.insert(0, '/root/codes/PTMM')

def calculate_latency(alpha, beta, pred):
    GRAPH = graph.get_graph()
    total_flops = 0
    for node in GRAPH:
        if node['type'] == 'conv':
            total_flops += node['flops']
    T = 0
    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            if alpha[node['conv_id']] == 0:
                if node['from'][0] != -1:
                    kernel_size, _, input_channels, output_channels = node[
                        'weights_shape']
                    parent_node = GRAPH[node['from'][0]]
                    if parent_node['type'] == 'add':
                        parent_parent_node_0 = GRAPH[parent_node['from'][0]]
                        parent_parent_node_1 = GRAPH[parent_node['from'][1]]
                        assert beta[parent_parent_node_0['conv_id']] == beta[
                            parent_parent_node_1['conv_id']]
                        input_channels_after_prune = int(
                            parent_parent_node_0['weights_shape'][3] *
                            (1 - beta[parent_parent_node_0['conv_id']]))
                    else:
                        input_channels_after_prune = 0
                        for i in node['from']:
                            parent_node = GRAPH[i]
                            input_channels_after_prune += int(
                                parent_node['weights_shape'][3] *
                                (1 - beta[parent_node['conv_id']]))
                    T_i = (1 - beta[node['conv_id']]) * (
                        input_channels_after_prune /
                        input_channels) * node['flops'] / total_flops
                else:
                    T_i = (1 -
                           beta[node['conv_id']]) * node['flops'] / total_flops
            else:
                kernel_size, _, input_channels, output_channels = node[
                    'weights_shape']
                if node['from'][0] != -1:
                    input_channels_after_prune = 0
                    parent_node = GRAPH[node['from'][0]]
                    if parent_node['type'] == 'add':
                        parent_parent_node_0 = GRAPH[parent_node['from'][0]]
                        parent_parent_node_1 = GRAPH[parent_node['from'][1]]
                        assert beta[parent_parent_node_0['conv_id']] == beta[
                            parent_parent_node_1['conv_id']]
                        input_channels_after_prune = int(
                            parent_parent_node_0['weights_shape'][3] *
                            (1 - beta[parent_parent_node_0['conv_id']]))
                    else:
                        for i in node['from']:
                            parent_node = GRAPH[i]
                            input_channels_after_prune += int(
                                parent_node['weights_shape'][3] *
                                (1 - beta[parent_node['conv_id']]))
                else:
                    input_channels_after_prune = input_channels
                
                output_channels_after_prune = int((1 - beta[node['conv_id']]) * output_channels)
                
                sparsity = alpha[node['conv_id']] / (1 - beta[node['conv_id']])
                T_i = (input_channels_after_prune / input_channels) * \
                      (output_channels_after_prune / output_channels) * \
                      node['flops'] / total_flops * \
                      pred.predict(output_channels_after_prune, kernel_size, kernel_size, input_channels, sparsity)
            T += T_i
    return T