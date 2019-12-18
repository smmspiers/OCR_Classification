"""
Classification system assignment solution

Samuel Spiers
COM3004 OCR Assignment
Version: v1.0
"""
import os
from urllib import request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eigh
from scipy.stats import mode
from scipy import ndimage
import utils.utils as utils


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
    cov_fvectors_train_full = np.cov(fvectors_train_full, rowvar=0)
    n = cov_fvectors_train_full.shape[0]
    v = eigh(cov_fvectors_train_full, eigvals=(n - 11, n - 2))[1]
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
        print('Applying gaussian filter to page', page_name.split('.')[1])
        images_train = [ndimage.gaussian_filter(image, 0.9) for image in images_train]
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    model_data['pca_matrix'] = learn_pca(fvectors_train_full)
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()

    print('Generating dictionaries of words for evaluation stage')
    model_data = generate_dictionaries(model_data)
    return model_data


def generate_dictionaries(model):
    """ Generates dictionary of words used for error correction

    :param model: dictionary, stores the output of the training stage
    :return: model: dictionary, stores the output of the training stage
    """
    dictionary = str(request.urlopen("http://www.mieliestronk.com/corncob_lowercase.txt").read()).split(r"\r\n")
    model['dictionary'] = dictionary

    current_dir = os.path.dirname(__file__)
    prob_dictionary_rel_file_path = "data/word_frequencies.txt"
    with open(os.path.join(current_dir, prob_dictionary_rel_file_path), "r") as prob_dictionary_file:
        prob_dictionary = [get_word_with_prob(line) for line in prob_dictionary_file]
        model['prob_dictionary'] = prob_dictionary
        prob_dictionary_file.close()
    return model


def get_word_with_prob(line):
    """ Gets word and corresponding probability from line of file

    :param line: A line in my word frequency text file
    :return: tuple of the word and its corresponding probability
    """
    line = line.split()
    return line[1].replace("'", "’"), int(line[2]) / 29213800


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
    """Classifier. Currently I am calling my k nearest neighbour
    classifier with k = 9.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    return k_nearest_neighbour_classifier(page, model, 9)


def nearest_neighbour_classifier(page, model):
    """Performs nearest neighbour classification

    :param page: matrix, each row is a feature vector to be classified
    :param model: dictionary, stores the output of the training stage
    :return: output_label
    """
    fvectors_train = np.array(model['fvectors_train'])
    fvectors_test = page
    labels_train = np.array(model['labels_train'])
    fvectors_train_mag = np.linalg.norm(fvectors_train, axis=1)
    fvectors_test_mag = np.linalg.norm(fvectors_test, axis=1)
    dot_product = np.dot(fvectors_test, fvectors_train.transpose())
    distances = dot_product / np.outer(fvectors_test_mag, fvectors_train_mag.transpose())
    nearest_neighbour = np.argmax(distances, axis=1)
    return labels_train[nearest_neighbour]


def k_nearest_neighbour_classifier(page, model, k):
    """Performs k nearest neighbour classification.

    :param page: matrix, each row is a feature vector to be classified
    :param model: dictionary, stores the output of the training stage
    :param k: number of nears neighbours
    :return: output_label
    """
    fvectors_train = np.array(model['fvectors_train'])
    fvectors_test = page
    labels_train = np.array(model['labels_train'])
    fvectors_train_mag = np.linalg.norm(fvectors_train, axis=1)
    fvectors_test_mag = np.linalg.norm(fvectors_test, axis=1)
    dot_product = np.dot(fvectors_test, fvectors_train.transpose())
    distances = dot_product / np.outer(fvectors_test_mag, fvectors_train_mag.transpose())
    k_nearest_distances = np.argsort(-distances, axis=1)[:, :k]
    nearest_neighbours = labels_train[k_nearest_distances]
    nearest_neighbour = mode(nearest_neighbours, axis=1)[0]
    return nearest_neighbour[:, 0]


def correct_errors(_, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    dictionary = model['dictionary']
    prob_dictionary = model['prob_dictionary']

    words = get_words(labels, bboxes)
    words_no_punc = [(sum(len(word) for word in words[:i]), remove_punctuation(word)) for i, word in enumerate(words)]

    incorrect_words = [(l_up_to, word) for l_up_to, word in words_no_punc if word.lower() not in dictionary]
    corrections = [(l_up_to, get_correction(word, prob_dictionary)) for l_up_to, word in incorrect_words]

    corrected_labels = [get_correction_for_label(i, corrections, labels) for i in range(len(labels))]
    return np.array(corrected_labels)


def get_correction_for_label(label_index, corrections, labels):
    """ Corrects a label by given label index.

    :param label_index: index of label to be corrected
    :param corrections: all corrections given to words
    :param labels: the output classification label for each feature vector
    :return: correct label if there is one
    """
    for l_up_to, correction in corrections:
        for j, char in enumerate(correction):
            if label_index == l_up_to + j:
                return char
    return labels[label_index]


def get_correction(word, dictionary):
    """ Get correction of a word

    :param word: word to be corrected
    :param dictionary: dictionary of real words for edits of word ti be tested against
    :return: corrected word
    """
    possible_corrections = [(real, prob) for real, prob in dictionary if len(real) == len(word)]
    edits = get_edits(word)
    possible_corrections2 = [(poss, prob) for poss, prob in possible_corrections if poss in edits]
    if len(possible_corrections2) == 0:
        return ""
    if len(possible_corrections2) == 1:
        return possible_corrections2[0][0]
    return max(possible_corrections2, key=lambda x: x[1])[0]


def get_edits(word):
    """ Generate possible edits of a word

    :param word: word to generate edits for
    :return: set of possible edits of a word
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz’"
    edits = set()
    for i in range(len(word)):
        for letter in alphabet:
            edit = list(word)
            edit[i] = letter
            edits.add(''.join(edit))
    return edits


def remove_punctuation(word):
    """Removes punctuation from word

    :param word: string to be depunctuated
    :return: word with no punctuation
    """
    punctuation = "!,-.;?"
    return ''.join(char for char in word if char not in punctuation)


def get_words(labels, bboxes):
    """Forms array of all the words on a page

    :param labels: the output classification label for each feature vector
    :param bboxes: 2d array, each row gives the 4 bounding box coords of the character
    :return: array of strings of each word for a page
    """
    spaces = [bboxes[i + 1, 0] - bboxes[i, 2] for i in range(bboxes.shape[0] - 1)]
    spaces.insert(0, 0)
    word = ""
    words = []
    for label, space in zip(labels, spaces):
        if space > 6 or space < -10:
            words.append(word)
            word = ""
        word += label
    return words


def search_google(word):
    """ Searches google for word and checks for 'Did you mean:'. However, since
    the evaluation stage is not run on a computer with an internet connection
    this won't work.

    :param word: word to be searched on google
    """
    url = 'https://www.google.com/search?q=' + word
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = request.Request(url, headers=headers)
    with request.urlopen(req) as response:
        html = response.read().decode("utf8")
        print('Word incorrectly spelt: ', 'Did you mean:' in html)
