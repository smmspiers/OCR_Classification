"""
Classification system assignment solution

Samuel Spiers
COM3004 OCR Assignment
Version: v1.0
"""
import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eigh


def plot_letter_image(letter, feature_vectors_full, model):
    """Method that plots the image of a letter

        Params:
        feature_vectors_full - feature vectors stored as rows
           in a matrix
        model - a dictionary storing the outputs of the model
           training stage
    """
    labels_train = np.array(model['labels_train'])
    letter_image = np.reshape(feature_vectors_full[labels_train == letter, :][0], (60, 39), order='F')
    plt.matshow(np.rot90(np.flip(letter_image, 0), -1), cmap=cm.Greys_r)
    plt.show()


def learn_pca(fvectors_train_full):
    """Generates PCA matrix

    This matrix is used in the linear transform PCA performs.

    :param fvectors_train_full: feature vectors stored as rows in a matrix
    :return PCA matrix
    """
    print(fvectors_train_full.shape)
    covx = np.cov(fvectors_train_full, rowvar=0)
    print("covx shape", covx.shape)
    n = covx.shape[0]
    w, v = eigh(covx, eigvals=(n - 10, n - 1))
    pca_matrix = np.fliplr(v)
    return pca_matrix.tolist()


def reduce_dimensions(feature_vectors_full, model):
    """Performs principal component analysis by reducing to 10 dimensions

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    pca_matrix = np.array(model['pca_matrix'])
    print(pca_matrix)
    pca_feature_vectors = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), pca_matrix)
    return pca_feature_vectors


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    # print('Reducing to 10 dimensions')
    model_data['pca_matrix'] = learn_pca(fvectors_train_full)
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    return np.repeat(labels_train[0], len(page))


def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    return labels
