import tflite
from tflite.BuiltinOperator import BuiltinOperator as OpType
from tflite.TensorType import TensorType
from tflite.Model import ModelT

import numpy as np
import flatbuffers
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=argparse.FileType(
        'rb'), help='Tflite model file to compress.')
    parser.add_argument('--output', '-o', type=argparse.FileType('wb'),
                        help='TFlite model file to write compressed result.')
    parser.add_argument('--cppmodel', '-c', type=argparse.FileType('w+'))
    parser.add_argument('--threshold', '-t', type=float, default=0.1)
    return parser.parse_args()


def load_model(fh):
    buf = bytearray(fh.read())
    model = tflite.Model.Model.GetRootAsModel(buf, 0)
    return tflite.Model.ModelT.InitFromObj(model)


def build_model(model):
    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b))
    return b.Output()


def save_model(buf, fh):
    fh.write(buf)


def tensor_size(model, subgraph, tensor):
    data = model.buffers[subgraph.tensors[tensor].buffer].data
    if data is None:
        return 0
    return len(data)


def model_size(model):
    weight_sizes = []
    bias_sizes = []
    opcodes = []

    s = model.subgraphs[0]
    for op in s.operators:
        if len(op.inputs) > 3:
            print('Unsupport node type.')

        if len(op.inputs) == 2:
            ws = tensor_size(model, s, op.inputs[1])
            weight_sizes.append(ws)
            bias_sizes.append(0)
        elif len(op.inputs) == 3:
            ws = tensor_size(model, s, op.inputs[1])
            bs = tensor_size(model, s, op.inputs[2])
            weight_sizes.append(ws)
            bias_sizes.append(bs)
        else:
            weight_sizes.append(0)
            bias_sizes.append(0)
        opcodes.append(op.opcodeIndex)
    return np.array(weight_sizes), np.array(bias_sizes), np.array(opcodes)


def compress(model, threshold=0.5):
    s = model.subgraphs[0]
    graph_ops = model.operatorCodes

    for op in s.operators:
        op_type = graph_ops[op.opcodeIndex]
        if op_type.deprecatedBuiltinCode == OpType.CONV_2D:
            t = op.inputs[1]
            ouput_ch, kernel_h, kernel_w, input_ch = s.tensors[t].shape
            tensor_weights = model.buffers[s.tensors[t].buffer].data.reshape(
                s.tensors[t].shape).astype(np.int8)
            sparsity = 0
            for x in range(ouput_ch):
                for y in range(kernel_h):
                    for z in range(kernel_w):
                        if np.all(tensor_weights[x, y, z, :] == 0):
                            sparsity += 1
            # sparsity = np.sum(tensor_weights == 0) / tensor_weights.size
            sparsity /= (ouput_ch * kernel_h * kernel_w)
            if sparsity < threshold:
                continue
            else:
                print('compress:', sparsity)
                compressed_tensor_weights = {
                    'values': [], 'col_idx': [], 'row_ptr': []}
                compressed_tensor_weights['row_ptr'].append(0)
                l = 0
                for i in range(ouput_ch):
                    for j in range(kernel_h):
                        for k in range(kernel_w):
                            if np.sum(np.abs(tensor_weights[i, j, k, :])) != 0:
                                l += 1
                                compressed_tensor_weights['values'].append(
                                    tensor_weights[i, j, k, :])
                                compressed_tensor_weights['col_idx'].append(
                                    (j * kernel_w + k) * input_ch)
                    compressed_tensor_weights['row_ptr'].append(l)
                if len(compressed_tensor_weights['values']) != 0:
                    compressed_tensor_weights['values'] = np.concatenate(
                        compressed_tensor_weights['values']).astype(np.int8)
                    compressed_tensor_weights['col_idx'] = np.array(
                        compressed_tensor_weights['col_idx']).astype(np.int16)
                else:
                    compressed_tensor_weights['values'] = np.array([0]).astype(np.int8)
                    compressed_tensor_weights['col_idx'] = np.array([0]).astype(np.int8)
                compressed_tensor_weights['row_ptr'] = np.array(
                    compressed_tensor_weights['row_ptr']).astype(np.int16)

                sparsity = tflite.SparsityParameters.SparsityParametersT()
                compressed_sparse_row = tflite.CompressedSparseRow.CompressedSparseRowT()
                compressed_sparse_row.rowPtr = compressed_tensor_weights['row_ptr']
                compressed_sparse_row.colIdx = compressed_tensor_weights['col_idx']
                compressed_sparse_row.inputCh = input_ch
                sparsity.compressedSparseRow = compressed_sparse_row
                model.buffers[s.tensors[t].buffer].data = compressed_tensor_weights['values']
                s.tensors[t].sparsity = sparsity


def save_c_array(model, fh):
    file_contents = "alignas(8) const unsigned char tflite_model[] = {\r\n"
    line_length = 12
    num_lines = int(np.floor(len(model)/line_length))
    for line in range(num_lines):
        line_content = "  " + ", ".join(["0x{:02x}".format(b) for b in model[line*line_length:(line+1)*line_length]])
        line_content += ",\r\n"
        file_contents += line_content
    file_contents += "  " + ", ".join(["0x{:02x}".format(b) for b in model[num_lines*line_length:]]) + "\r\n};\r\n"
    file_contents += "const int tflite_model_len = {};".format(len(model))
    fh.write(file_contents)


if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.input)
    weight_sizes, bias_sizes, _ = model_size(model)
    print("Model size before compression - Weights: {:.2f} KiB, Biases: {:.2f} KiB".format(
        np.sum(weight_sizes)/2**10, np.sum(bias_sizes)/2**10))
    compress(model, args.threshold)
    buf = build_model(model)

    save_model(buf, args.output)
    save_c_array(buf, args.cppmodel)
