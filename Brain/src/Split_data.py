import os
import random
import shutil


def split_data(source_directory, train_directory, test_directory, train_ratio=0.8):
    for folder in [train_directory, test_directory]:
        if not os.path.exists(folder):
            os.makedirs(os.path.join(folder, 'Normal'))
            os.makedirs(os.path.join(folder, 'Stroke'))

    categories = ['Normal', 'Stroke']
    for category in categories:
        category_path = os.path.join(source_directory, category)
        files = [os.path.join(category_path, f) for f in os.listdir(category_path) if
                 os.path.isfile(os.path.join(category_path, f))]

        random.shuffle(files)

        train_size = int(len(files) * train_ratio)
        train_files = files[:train_size]
        test_files = files[train_size:]

        train_category_path = os.path.join(train_directory, category)
        for file in train_files:
            shutil.copy(file, train_category_path)

        test_category_path = os.path.join(test_directory, category)
        for file in test_files:
            shutil.copy(file, test_category_path)


source_directory = 'D:\\Code\\Vigilant-VGG16\\Brain\\resoucre\\data'
train_directory = 'D:\\Code\\Vigilant-VGG16\\Brain\\resoucre\\train'
test_directory = 'D:\\Code\\Vigilant-VGG16\\Brain\\resoucre\\test'

split_data(source_directory, train_directory, test_directory)
