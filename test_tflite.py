from utils import *
import numpy as np
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    
    interpreter = tf.lite.Interpreter(args.model_path)
    input = interpreter.get_input_details()[0]
    output = interpreter.get_output_details()[0]
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    input_quant_param = input_details['quantization_parameters']
    scale, zero_point = input_quant_param['scales'][0], input_quant_param['zero_points'][0]
    
    if args.dataset == 'cifar10':
        image_size = 32
    elif args.dataset == 'vww':
        image_size = 64
    train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch =get_train_and_test_generator(image_size, 1, args.dataset, 0.2)
    
    accuracy = 0.
    for i in tqdm(range(test_generator.__len__())):
        x, y_gt = test_generator.__getitem__(i)
        y_gt = y_gt[0]
        x = x / scale + zero_point
        x = x.astype(input_details['dtype'])
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output['index'])[0]
        if np.argmax(y_pred) ==  np.argmax(y_gt):
            accuracy += 1
    
    print(accuracy / test_generator.__len__())
    
    