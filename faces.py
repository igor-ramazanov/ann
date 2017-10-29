#!/usr/local/bin/python3.6

"""
Trains an ANN on training Images and prints classification results for testing Images.

Using:
python main.py \
<path to file with training images> \
<path to file with answers for training images> \
<path to file with testing images>
"""
from functools import reduce
import sys
import numpy as np

LEARNING_RATE = 0.005
PRECISION_THRESHOLD = 0.68
TRAINING_SET_PERCENTAGE = 0.7


class Image:
    def __init__(self, id):
        self.id = id
        self.pixels = np.array([])
        self.answer = -1


def read_images(file_name):
    """
    Read images file and create a list of images.

    Arguments:
        file_name (string): path to file with images.

    Returns:
        list: a list of Image class instances.
    """
    with open(file_name) as f:
        lines = [line[:-1] for _, line in enumerate(f)]

    without_comments = [line for line in lines if not line.startswith('#')]
    without_empty_lines = [line for line in without_comments if len(line) != 0]
    images = []
    for line in without_empty_lines:
        if line.startswith("Image"):
            image = Image(id=line)
            images.append(image)
        else:
            pixels = np.array([np.float64(n) / np.float64(31.0) for n in line.split(" ")])
            image = images[-1]
            image.pixels = np.concatenate([image.pixels, pixels])
    return images


def convert_answer_to_array(answer):
    """
    Converts an answer (emotion of an image) from number to array.

    Arguments:
        answer (int): emotion code, should be in range: [1, 4]

    Returns:
        np.array: a NumPy array of emotions where proper emotion equals 1 and others 0
    """
    array = np.array([0, 0, 0, 0])
    array[answer - 1] = 1
    return array


def convert_array_to_answer(answer_array):
    """
    Converts an answer array from `convert_answer_to_array` function back to number.

    Arguments:
        answer_array (np.array): array of coded emotions like: [0, 0, 0, 1].

    Returns:
        int: number code of an emotion.
    """
    i, = np.where(answer_array == 1)
    return i[0] + 1


def read_answers(images, file_name):
    """
    Fulfill an images list with answers from an answers file.

    Arguments:
        images (list): a list of Image class instances.
        file_name (string): path to file with emotions for each image.

    Returns:
        void: Returns nothing
    """
    with open(file_name) as f:
        lines = [line[:-1] for _, line in enumerate(f)]

    without_comments = [line for line in lines if not line.startswith('#')]
    without_empty_lines = [line for line in without_comments if len(line) != 0]
    for line in without_empty_lines:
        image_id, answer = line.split(" ")
        image = next(x for x in images if x.id == image_id)
        image.answer = convert_answer_to_array(int(answer))


def train(training_set, testing_set, learning_rate):
    """
    Trains on training set of images and returns `prediction_function` for image classification.

    Arguments:
        training_set (list): a list of Image class instances.
        learning_rate (float): coefficient of learning speed of Artificial Neural Network.
        iterations_count (int): number of iterations for learning, each iteration processes each image only once

    Returns:
        function: prediction function that receives an Image and returns emotion code.
    """
    weights_matrix = np.random.random((400, 4))

    def calc_current_output(image):
        weight_sums = np.dot(image.pixels, weights_matrix)
        return 1.0 / (1.0 + np.exp(-weight_sums))

    def prediction_function(image):
        output = calc_current_output(image)
        return np.argmax(output) + 1

    def is_good_enough():
        return measure_precision(prediction_function, testing_set) > PRECISION_THRESHOLD

    while True:
        for image in training_set:
            if is_good_enough():
                return prediction_function
            current_output = calc_current_output(image)
            error_layer = image.answer - current_output
            weights_deltas_matrix = np.reshape(image.pixels, (400, 1)) * error_layer * learning_rate
            weights_matrix += weights_deltas_matrix


def measure_precision(f, testing_set):
    """
    Measures precision of `prediction_function` on a testing set of Images.
    Precision is calculated as percent of properly classified Images from testing set.

    Arguments:
        f: (function): `prediction function` that receives an Image and returns emotion code.
        testing_set (list): a list of Image class instances

    Returns:
        float: Percent of properly classified Images, changes from [0.0, 1.00]
    """

    def is_right(image):
        predicted_answer = f(image)
        return int(predicted_answer == convert_array_to_answer(image.answer))

    count_of_right = reduce(lambda acc, image: acc + is_right(image), testing_set, 0.0)
    return count_of_right / len(testing_set)


def print_answers(images, prediction_function):
    for image in images:
        print("%s %d" % (image.id, prediction_function(image)))


if __name__ == '__main__':
    training_images = read_images(sys.argv[1])
    read_answers(training_images, sys.argv[2])
    testing_images = read_images(sys.argv[3])

    images_count = len(training_images)
    training_set_size = int(images_count * TRAINING_SET_PERCENTAGE)
    training_set = training_images[0:training_set_size]
    testing_set = training_images[training_set_size:]

    prediction_function = train(training_set, testing_set, LEARNING_RATE)
    print_answers(testing_images, prediction_function)
