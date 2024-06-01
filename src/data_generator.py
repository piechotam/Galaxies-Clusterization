import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from processing import process_image

class ImageDataGenerator(Sequence):
    """
    ImageDataGenerator is a custom data generator for image data that yields batches of images for training a model.
    This generator reads image files, processes them (e.g., resizing and normalization), and supplies them to the model
    in batches.
    
    Attributes
    ----------
    image_filenames : list
        A list of file paths to the images.
    batch_size : int
        The number of images in each batch.
    img_size : int
        The size (height and width) to which each image will be resized.
        
    Methods
    -------
    __len__()
        Returns the number of batches per epoch.
    __getitem__(idx)
        Generates one batch of data.
    """

    def __init__(self, image_filenames, batch_size, img_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.image_filenames[i] for i in indices]
        images = [process_image(filename, 20, (self.img_size, self.img_size)) for filename in batch_x]
        images = np.array(images)
        images = np.expand_dims(images, axis=-1)

        return images, images
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_filenames))
        np.random.shuffle(self.indices)