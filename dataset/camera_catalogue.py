import tensorflow as tf
import os
import pandas as pd
import shutil


def split_trainset_and_testset(root_path, size=0.3):
    data = []
    wo = 0
    w = 0
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if folder == 'blank' or folder == 'vehicle':
                data.append([img_path, 0, 0])
                wo += 1
            else:
                data.append([img_path, 1, 0])
                w += 1
    print(w)
    print(wo)
    df = pd.DataFrame(data, columns=['image_path', 'label', 'test'])
    df.sample(frac=1).reset_index(drop=True)
    df.loc[df.sample(frac=size).index, 'test'] = 1

    split_csv_path = os.path.join(root_path, 'split.csv')
    df.to_csv(split_csv_path)


def build_trainset_and_testset(root_path, split_csv_name):
    split_csv_path = os.path.join(root_path, split_csv_name)
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)

    train_animal_path = os.path.join(train_path, 'animal')
    train_non_animal_path = os.path.join(train_path, 'non_animal')
    test_animal_path = os.path.join(test_path, 'animal')
    test_non_animal_path = os.path.join(test_path, 'non_animal')

    os.mkdir(train_animal_path)
    os.mkdir(train_non_animal_path)
    os.mkdir(test_animal_path)
    os.mkdir(test_non_animal_path)

    df = pd.read_csv(split_csv_path)

    train_animal_df = df[(df['test'] == 0) & (df['label'] == 1)]
    train_non_animal_df = df[(df['test'] == 0) & (df['label'] == 0)]
    test_animal_df = df[(df['test'] == 1) & (df['label'] == 1)]
    test_non_animal_df = df[(df['test'] == 1) & (df['label'] == 0)]

    for index, row in train_animal_df.iterrows():
        shutil.copy(row['image_path'], train_animal_path)
    for index, row in train_non_animal_df.iterrows():
        shutil.copy(row['image_path'], train_non_animal_path)
    for index, row in test_animal_df.iterrows():
        shutil.copy(row['image_path'], test_animal_path)
    for index, row in test_non_animal_df.iterrows():
        shutil.copy(row['image_path'], test_non_animal_path)


def get_camera_catalogue_train_and_test_generator(root_path, image_size, batch_size, val=0.0):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        rescale=1. / 255,
        validation_split=val)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_path = os.path.join(root_path, 'train')
    train_size = len(os.listdir(os.path.join(train_path, 'animal'))) + len(os.listdir(os.path.join(train_path, 'non_animal')))
    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        subset='training')
    val_generator = datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        subset='validation')

    test_path = os.path.join(root_path, 'test')
    test_size = len(os.listdir(os.path.join(test_path, 'animal'))) + len(os.listdir(os.path.join(test_path, 'non_animal')))
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb')
    
    return train_generator, val_generator, test_generator, int(train_size * (1 - val)) // batch_size, int(train_size * val) // batch_size, test_size // batch_size

if __name__ == '__main__':
    # split_trainset_and_testset('/data/camera_catalogue_images/camera_catalogue/all')
    build_trainset_and_testset('/data/camera_catalogue_images/camera_catalogue/all', 'split.csv')