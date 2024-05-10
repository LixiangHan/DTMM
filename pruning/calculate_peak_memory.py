import numpy as np
import graph


def calculate_peak_memory(beta, sram_constrain):
    GRAPH = graph.get_graph()

    def calculate_current_memory(tensors_list):
        mem = 0
        for tensor in tensors_list:
            mem += tensor[1]
        return mem

    def get_bottlenecks_idx(tensors_list):
        b = []
        for tensor in tensors_list:
            if tensor[0] != -1 and GRAPH[tensor[0]]['type'] == 'conv':
                b.append(GRAPH[tensor[0]]['conv_id'])
        return b

    def free_memory(tensors_list, after):
        for i, tensor in enumerate(tensors_list):
            not_relied = True
            for node in GRAPH[after:]:
                if tensor[0] in node['from']:
                    not_relied = False
                    break
            if not_relied:
                tensors_list.pop(i)
        return tensors_list

    peak_memory = 0
    bottlenecks_conv_idx = []
    tensors_in_memory = []  # [[id, size], ...]

    for node in GRAPH:
        tensors_in_memory = free_memory(tensors_in_memory, node['id'])

        if node['from'][0] != -1:
            if node['type'] == 'conv' and node['prune'] == True:
                output_width, _, output_channels = node['output_shape']
                output_channels_after_prune = int(output_channels *
                                                  (1 - beta[node['conv_id']]))
                output_size = np.prod(
                    [output_width, output_width, output_channels_after_prune])
                tensors_in_memory.append([node['id'], output_size])
            elif node['type'] == 'add':
                output_width, _, output_channels = GRAPH[node['from'][0]]['output_shape']
                parent_node_0 = GRAPH[node['from'][0]]
                parent_node_1 = GRAPH[node['from'][1]]
                assert beta[parent_node_0['conv_id']] == beta[parent_node_1['conv_id']]
                output_channels_after_prune = int(
                    output_channels * int(1 - beta[parent_node_0['conv_id']]))
                output_size = np.prod(
                    [output_width, output_width, output_channels_after_prune])
                tensors_in_memory.append([node['id'], output_size])
            elif node['type'] == 'fc':
                # not important
                pass
            elif node['type'] == 'conv' and node['prune'] == False:
                output_width, _, output_channels = node['output_shape']
                output_channels_after_prune = output_channels
                output_size = np.prod(
                    [output_width, output_width, output_channels_after_prune])
                tensors_in_memory.append([node['id'], output_size])
            else:
                print(node['type'], 'is not supported yet!')
        else:
            input_size = np.prod(node['input_shape'])
            tensors_in_memory.append([node['from'][0], input_size])
            output_width, _, output_channels = node['output_shape']
            output_channels_after_prune = int(output_channels *
                                              (1 - beta[node['conv_id']]))
            output_size = np.prod(
                [output_width, output_width, output_channels_after_prune])
            tensors_in_memory.append([node['id'], output_size])
        current_memory = calculate_current_memory(tensors_in_memory)
        if current_memory > peak_memory:
            peak_memory = current_memory
            if peak_memory > sram_constrain:
                bottlenecks_conv_idx = get_bottlenecks_idx(tensors_in_memory)
            else:
                bottlenecks_conv_idx = []
                
    return peak_memory, bottlenecks_conv_idx