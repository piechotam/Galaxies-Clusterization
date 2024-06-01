import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split

def find_extensions(path):
    extensions = {}

    with os.scandir(path) as files:
        for file in files:
            filename, ext = os.path.splitext(file.name)
            if ext not in extensions:
                extensions[ext] = [filename]
            else:
                extensions[ext].append(filename)

    return extensions


def read_filenames(path):
    galaxies = []

    with os.scandir(path) as files:
        for file in files:
            if file.name.endswith('.jpg'):
                galaxies.append(file.name)
    
    return galaxies

def prepare_datasets(dataset, size):
    if size > len(dataset):
        raise ValueError(f'Size too large, dataset contains {len(dataset)} observations.')
    random.seed(21)
    random.shuffle(dataset)
    dataset = dataset[:size]
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=21)
    dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.2, random_state=21)

    return dataset_train, dataset_val, dataset_test

def show_galaxies(galaxies, rand = False):
    figure, axes = plt.subplots(nrows=5,ncols=5,figsize=(13,13))
    for i, ax in enumerate(axes.flat):
        indx = random.randint(0, len(galaxies) - 1) if rand else i
        img = plt.imread(galaxies[indx])
        ax.imshow(img)
        ax.set_xlabel(f'Galaxy {indx}')
    plt.tight_layout()
    plt.show()