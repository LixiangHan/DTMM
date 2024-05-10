from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--flash', type=int)
    parser.add_argument('--sram', type=int)
    parser.add_argument('--acc', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args