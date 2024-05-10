import numpy as np
from matplotlib import pyplot as plt

from calculate_latency import calculate_latency
from calculate_model_size import calculate_model_size
from calculate_peak_memory import calculate_peak_memory
import graph


def print_array(arr, label=None):
    if label != None:
        print(label, end=':\n')
    for a in arr:
        print('%.5f\t' % a, end=' ')
    print('\n----------------------------------------')

def solve_sa(init_alpha,
             init_beta,
             objective_function,
             filterlet_importance,
             filter_importance,
             flash_constrain,
             sram_constrain,
             accuracy_constrain,
             predictor,
             T_0=100,
             T_t=0.1,
             k=0.9,
             L=100,
             disturbance=0.05):
    
    GRAPH = graph.get_graph()
    def align(alpha, beta):
        for i in range(len(GRAPH)):
            if GRAPH[i]['type'] == 'conv' and GRAPH[i]['prune'] == True  and GRAPH[i]['weights_shape'][0] == 1:
                conv_id = GRAPH[i]['conv_id']
                beta[conv_id] += alpha[conv_id]
                alpha[conv_id] = 0
            elif GRAPH[i]['type'] == 'add':
                parent_node_1 = GRAPH[GRAPH[i]['from'][0]]
                parent_node_2 = GRAPH[GRAPH[i]['from'][1]]
                parent_node_1_conv_id = parent_node_1['conv_id']
                parent_node_2_conv_id = parent_node_2['conv_id']
                beta[parent_node_1_conv_id] = beta[parent_node_2_conv_id] = min(beta[parent_node_1_conv_id], beta[parent_node_2_conv_id])
        return alpha, beta
    
    
    def align_margin(margin):
        for i in range(len(GRAPH)):
            if GRAPH[i]['type'] == 'add':
                parent_node_1 = GRAPH[GRAPH[i]['from'][0]]
                parent_node_2 = GRAPH[GRAPH[i]['from'][1]]
                parent_node_1_conv_id = parent_node_1['conv_id']
                parent_node_2_conv_id = parent_node_2['conv_id']
                margin[parent_node_1_conv_id] = margin[parent_node_2_conv_id] = min(margin[parent_node_1_conv_id], margin[parent_node_2_conv_id])
        return margin
                


    def check_accuracy_constrain(alpha, beta, alpha_bound, beta_bound):
        beta = beta.clip(0, beta_bound)
        alpha = alpha.clip(0, alpha_bound - beta)
        return alpha, beta

    T = T_0
    num_layers = init_alpha.size
    
    alpha_bound = np.array([
        np.sum(filterlet_importance[i] <= accuracy_constrain) /
        filterlet_importance[i].size for i in range(num_layers)
    ])
    
    beta_bound = np.array([
        np.sum(filter_importance[i] <= accuracy_constrain) /
        filter_importance[i].size for i in range(num_layers)
    ])
    
    print('>>>', alpha_bound)
    print('>>>', beta_bound)
    
    best_obj = float('inf')
    best_alpha = init_alpha[:]
    best_beta = init_beta[:]
    
    current_obj = float('inf')
    current_alpha = init_alpha[:]
    current_beta = init_beta[:]
    
    
    obj_list = []
    while (T > T_t):
        print(T)
        for i in range(L):
            delta_alpha = np.random.random(
                num_layers
            ) * disturbance * 2 - disturbance  # delta_alpha in (-disturbance, dicturbance)
            delta_beta = np.random.random(
                num_layers) * disturbance * 2 - disturbance
            new_alpha = current_alpha + delta_alpha
            new_beta = current_beta + delta_beta
            
            new_alpha, new_beta = check_accuracy_constrain(new_alpha, new_beta, alpha_bound, beta_bound)
            new_alpha, new_beta = align(new_alpha, new_beta)

            peak_memory, bottlenecks_idx = calculate_peak_memory(new_beta, sram_constrain)

            while peak_memory > sram_constrain:
                margin = beta_bound - new_beta
                margin = align_margin(margin)
                if np.any(margin[bottlenecks_idx] > 0):
                    for i in bottlenecks_idx:
                        if margin[i] > 0:
                            new_beta[i] += min(0.05, margin[i])
                else:
                    new_beta[bottlenecks_idx] += 0.01
                    new_beta = new_beta.clip(max=0.8)
                    
                new_alpha, new_beta = align(new_alpha, new_beta)
                peak_memory, bottlenecks_idx = calculate_peak_memory(new_beta, sram_constrain)

            model_size = calculate_model_size(new_alpha, new_beta)
            while model_size > flash_constrain:
                # print_array(new_beta, 'new beta')
                is_enough = False
                # increase beta
                margin_beta = beta_bound - new_beta
                margin_beta = align_margin(margin_beta)
                # print_array(new_beta, 'new beta')
                
                # print_array(margin_beta, 'margin beta')
                
                if np.any(margin_beta > 1e-4):
                    for i in np.where(margin_beta > 1e-4)[0]:
                        new_beta[i] += min(0.05, margin_beta[i])
                    is_enough = True
                # print_array(new_beta, 'new beta')
                
                margin_alpha = alpha_bound - (new_alpha + new_beta)
                new_alpha += margin_alpha.clip(max=0)
                new_alpha = new_alpha.clip(min=0)
                margin_alpha = margin_alpha.clip(min=0)
                
                # print_array(margin_alpha, 'margin alpha')
                
                for i in range(len(GRAPH)):
                    node = GRAPH[i]
                    if node['type'] == 'conv' and node['prune'] == True and node['weights_shape'][0] == 1:
                        # do not increase beta if this layer is 1x1
                        margin_alpha[node['conv_id']] = 0
                
                # print('>>>', margin_alpha)
                if np.any(margin_alpha > 1e-4):
                    for i in np.where(margin_alpha > 1e-4)[0]:
                        new_alpha[i] += min(0.05, margin_alpha[i])
                    is_enough = True
                
                if not is_enough:
                    new_beta[2:] += 0.01
                    new_beta = new_beta.clip(max=0.8)
        
                
                new_alpha, new_beta = align(new_alpha, new_beta)
                # print_array(new_beta, 'new beta')
                model_size = calculate_model_size(new_alpha, new_beta)

                # print(model_size)
                # input()

            new_obj = objective_function(new_alpha, new_beta, predictor)

            df = new_obj - current_obj
            if df < 0 or np.exp(-df / T) > np.random.rand():
                current_obj = new_obj
                current_alpha = new_alpha[:]
                current_beta = new_beta[:]
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_alpha = new_alpha[:]
                    best_beta = new_beta[:]
    
            obj_list.append(best_obj)
        T = k * T
    plt.plot(obj_list)
    plt.savefig('tmp.png')
    print(calculate_peak_memory(best_beta, sram_constrain)[0])
    print('objective: ', best_obj)
    return best_alpha, best_beta