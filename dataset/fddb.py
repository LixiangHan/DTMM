from tensorflow import keras
from matplotlib import pyplot as plt
import cv2
import numpy as np


classes_num = {'face': 0}

def random_flip(image, label):
    if np.random.random() > 0.5:
        image_h, image_w = image.shape[0:2]
        image = cv2.flip(image, 1)
        for i in range(len(label)):
            xmin, ymin, xmax, ymax, c = label[i]
            xmin = image_w - xmin
            xmax = image_w - xmax
            label[i] = (xmin, ymin, xmax, ymax, c)
    return image, label


def random_bright(image):
    if np.random.random() > 0.5:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        h, s, v = cv2.split(image)
        beta = np.random.randint(-64, 64)
        v = np.uint8(np.clip(cv2.add(1 * v, beta), 0, 255))
        image = np.uint8(cv2.merge((h,s,v)))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def random_contrast(image):
    if np.random.random() > 0.5:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        h, s, v = cv2.split(image)
        alpha = 0.5 + np.random.random() * (1.5 - 0.5)
        v = np.uint8(np.clip(cv2.add(alpha * v, 0), 0, 255))
        image = np.uint8(cv2.merge((h,s,v)))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


class VOCGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, image_shape=(448, 448), augmentation=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.augmentation = augmentation
    
    def __len__(self):
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        train_image = []
        train_label = []
        
        for image_path, label in zip(batch_x, batch_y):
            image, label_matrix = self.read(image_path, label)
            train_image.append(image)
            train_label.append(label_matrix)

        return np.array(train_image, dtype=np.float32), np.array(train_label, dtype=np.float32)
    
    def read(self, image_path, label):
        raise NotImplementedError
    

class FDDBGenerator(VOCGenerator):
    def __init__(self, images, labels, batch_size, image_shape=(448, 448), augmentation=False):
        super().__init__(images, labels, batch_size, image_shape, augmentation)

    
    def read(self, image_path, label):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_h, image_w = image.shape[0:2]
        

        parsed_label = []
        for l in label:
            l = l.split(',')
            l = np.array(l, dtype=np.int)
            xmin, ymin, xmax, ymax, c = l
            parsed_label.append((xmin, ymin, xmax, ymax, c))

        if self.augmentation:
            image, parsed_label = random_flip(image, parsed_label)
            image = random_bright(image)
            image = random_contrast(image)

        image = cv2.resize(image, self.image_shape) # shoud resize after augumentation
        image = image / 255.

        label_matrix = np.zeros([7, 7, 1 + 5 * 2])
        for l in parsed_label:
            xmin, ymin, xmax, ymax, c = l

            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h

            loc = [7 * x, 7 * y]
            loc_i = int(loc[0])
            loc_j = int(loc[1]) # The object belongs to cell(loc_i, loc_j)
            x = loc[0] - loc_i
            y = loc[1] - loc_j

            # The (x,y) coordinates represent the center of the box relative to the bounds of 
            # the grid cell. The width and height are predicted relative to the whole image
            # label_matrix: [[[cls1, cls2,... , clsn, x1, y1, w1, w1, conf1, ...], ...], ...]
            if label_matrix[loc_j, loc_i, 5] == 0: # confidence/response
                label_matrix[loc_j, loc_i, c] = 1
                # label_matrix[loc_j, loc_i, :] = smooth_labels(label_matrix[loc_j, loc_i, c])
                label_matrix[loc_j, loc_i, 1:5] = [x, y, w, h]
                label_matrix[loc_j, loc_i, 5] = 1
            
        return image, label_matrix

    
def load_fddb_dataset(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()

    X = []
    Y = []
    
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(' ')
        image_path = line[0]
        labels = line[1:]
        X.append(image_path)
        Y.append(labels)
    return X, Y