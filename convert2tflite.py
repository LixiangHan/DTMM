import tensorflow as tf
from utils.utils import convert_to_tflite
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    
    if args.dataset == 'cifar10':
        image_size = 32
    elif args.dataset == 'vww':
        image_size = 64
    
    convert_to_tflite(model, args.dataset, image_size, args.save_path)