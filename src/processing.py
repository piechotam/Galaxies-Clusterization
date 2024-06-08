import cv2 as cv
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def load_image(filename):
    """Given an image filename, reads the image"""
    return cv.imread(f'../data/images/{filename}')

def find_bounding_boxes(image, threshold):
    """Finds bounding boxes in image with provided threshold."""
    image = image.copy()
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(image_gray, threshold, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)

    return image

def find_main_object(image, threshold):
    """Finds main object (galaxy) in given image returns its contour"""
    image = image.copy()
    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_image = cv.threshold(image_gray, threshold, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    def calculate_contour_distance(c):
        x, y, w, h = cv.boundingRect(c)
        x_center = x + w / 2
        y_center = y + h / 2
       
        return np.sqrt(pow(x_center - image_center[0], 2) + pow(y_center - image_center[1], 2))

    return max(contours, key=lambda c: cv.contourArea(c) - pow(calculate_contour_distance(c), 3/2))
        
def extract_and_resize(image, threshold, target_size=(128, 128)):
    """Finds contour of main object and crops the image to extract only the main object.
    The image is cropped to a square so that after resizing it does not stretch."""
    contour = find_main_object(image, threshold)
    x, y, w, h = cv.boundingRect(contour)
    
    max_dim = max(w, h)
    x_center = x + w // 2
    y_center = y + h // 2

    x1 = max(0, x_center - 10 - max_dim // 2)
    y1 = max(0, y_center - 10 - max_dim // 2)
    x2 = min(image.shape[1], x_center + 10 + max_dim // 2)
    y2 = min(image.shape[0], y_center + 10 + max_dim // 2)

    img_cropped = image[y1:y2, x1:x2]
    resized_image = cv.resize(img_cropped, target_size, interpolation=cv.INTER_AREA)

    return resized_image

def convert_to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def normalize_image(image):
    """Normalizes the image."""
    return image / 255.0

def process_image(filename, threshold, target_size=(128, 128)):
    """Proccesses the image by loading it, extracting main object and normalizing."""
    image = load_image(filename)
    cropped_image = extract_and_resize(image, threshold, target_size)
    image_grayscale = convert_to_grayscale(cropped_image)
    processed_image = normalize_image(image_grayscale)

    return processed_image

def extract_features(image, model):
    img = np.array(image)
    img = np.stack((img,) * 3, axis=-1)
    reshaped_img = img.reshape(1, 224, 224, 3)
    features = model.predict(reshaped_img, verbose=0)
    return features

def extract_hog_features(image):
    resized_image = cv.resize(image, (64, 64), interpolation=cv.INTER_AREA)
    features, _ = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    
    return features

def convert_to_feature_vectors(filenames, autoencoder, img_size, hog=False, model=None, size=None):
    """
    Converting an image to a feature vector. If a model is provided, we use it during the proces.
    Otherwise, if hog = True we convert images to vector using HOG (Histogram of Oriented Gradients)
    Else, the image of size n by n is simply reshaped into vector of length n^2.
    """
    if not size:
        size = len(filenames)

    features = []

    for i, filename in enumerate(filenames[:size]):
        if i % 1000 == 999:
            print(f'{i + 1} out of {size}')
        image = process_image(filename, 20, img_size)
        image = autoencoder.predict(np.array([image]), verbose=0)[0]

        if model:
            image = cv.resize(image, (224, 224), cv.INTER_AREA)
            feat = extract_features(image, model)
        
        elif hog:
            feat = extract_hog_features(image)

        else:
            feat = image.reshape(-1)
        
        features.append(feat)


    features = np.array(features)
    features = features.reshape(features.shape[0], -1)

    return features

def analyse_pca(features):
    print('Scaling data...')
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print('Fitting pca with 700 components...')
    pca = PCA(n_components=700, random_state=21)
    pca.fit(features)

    plt.figure(figsize=(12, 6))
    xi = np.arange(0, 700, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 701, step=50)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(30, 0.9, '95% cut-off threshold', color = 'red', fontsize=16)

    plt.show()

    return pca

def find_n_comp(pca, percentage):
    n_components_percentage = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= percentage / 100) + 1
    print(f'Number of components for {percentage}% explainability: {n_components_percentage}')
    
    return n_components_percentage