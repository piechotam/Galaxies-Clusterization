import cv2 as cv
import numpy as np

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
    reshaped_img = img.reshape(1, 80, 80, 3)
    features = model.predict(reshaped_img, verbose=0)
    return features

def convert_to_feature_vectors(filenames, autoencoder, model, size=None):
    if not size:
        size = len(filenames)

    features = []

    for i, filename in enumerate(filenames[:size]):
        if i % 1000 == 999:
            print(f'{i + 1} out of {size}')
        image = process_image(filename, 20, (80, 80))
        image = autoencoder.predict(np.array([image]), verbose=0)[0]    
        feat = extract_features(image, model)
        features.append(feat)


    features = np.array(features)
    features = features.reshape(features.shape[0], -1)

    return features