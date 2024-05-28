import matplotlib.pyplot as plt
import random
import os

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


def show_galaxies(galaxies, rand = False):
    figure, axes = plt.subplots(nrows=5,ncols=5,figsize=(13,13))
    for i, ax in enumerate(axes.flat):
        indx = random.randint(0, len(galaxies) - 1) if rand else i
        img = plt.imread(galaxies[indx])
        ax.imshow(img)
        ax.set_xlabel(f'Galaxy {indx}')
    plt.tight_layout()
    plt.show()