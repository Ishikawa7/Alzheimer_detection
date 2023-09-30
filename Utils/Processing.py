import os
import numpy as np
from PIL import Image
import random

def class_image_occurences(dir_train: str = "Train/" , dir_test: str = "Test/") -> tuple[dict, dict, dict]:
    dict_train_count = {}
    dict_test_count = {}

    sum_train = 0
    sum_test = 0
    for folder in os.listdir(dir_train):
        count_train = len(os.listdir(dir_train+folder))
        count_test = len(os.listdir(dir_test+folder))
        sum_train += count_train
        sum_test += count_test
        dict_train_count[folder] = count_train
        dict_test_count[folder] = count_test
    dict_train_count["Total"] = sum_train
    dict_test_count["Total"] = sum_test

    return dict_train_count, dict_test_count

def class_weights(dir_train: str = "Train/") -> dict:
    # calculate class weights
    sum_train = 0
    class_weights = {}
    for folder_class_name in os.listdir(dir_train):
        count_train = len(os.listdir(dir_train+folder_class_name))
        sum_train += count_train
        class_weights[folder_class_name] = count_train

    for key in class_weights.keys():
        class_weights[key] = sum_train/class_weights[key]
    
    # normalize class weights
    max_weight = max(class_weights.values())
    min_weight = min(class_weights.values())
    for key in class_weights.keys():
        class_weights[key] = (class_weights[key]-min_weight)/(max_weight-min_weight)

    return class_weights

def train_numpy_arrays(class_weights: dict, dir_train :str = "Train/") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images_list_train = []
    labels_list_train = []
    class_weights_list_train = []

    for folder in os.listdir(dir_train):
        for image in os.listdir(dir_train+folder):
            image_file = Image.open(dir_train+folder+"/"+image)
            np_image = np.array(image_file)
            np_image = np_image/np_image.max() # normalize one by one
            image_file.close()
            images_list_train.append(np_image)
            labels_list_train.append(folder)
            class_weights_list_train.append(class_weights[folder])

    # shuffle
    combined_list = list(zip(images_list_train, labels_list_train, class_weights_list_train))
    random.shuffle(combined_list)
    images_list_train, labels_list_train, class_weights_list_train = zip(*combined_list)

    X_train = np.array(images_list_train)
    y_train = np.array(class_weights_list_train)
    y_train = y_train.reshape(-1,1)
    labels_train = np.array(labels_list_train)
    labels_train = labels_train.reshape(-1,1)

    return X_train, y_train, labels_train

def test_numpy_arrays(class_weights: dict, dir_test :str = "Test/") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images_list_test = []
    labels_list_test = []
    class_weights_list_test = []

    for folder in os.listdir(dir_test):
        for image in os.listdir(dir_test+folder):
            image_file = Image.open(dir_test+folder+"/"+image)
            np_image = np.array(image_file)
            np_image = np_image/np_image.max() # normalize one by one
            image_file.close()
            images_list_test.append(np_image)
            labels_list_test.append(folder)
            class_weights_list_test.append(class_weights[folder])

    # shuffle
    combined_list = list(zip(images_list_test, labels_list_test, class_weights_list_test))
    random.shuffle(combined_list)
    images_list_test, labels_list_test, class_weights_list_test = zip(*combined_list)
    
    X_test = np.array(images_list_test)
    y_test = np.array(class_weights_list_test)
    y_test = y_test.reshape(-1,1)
    labels_test = np.array(labels_list_test)
    labels_test = labels_test.reshape(-1,1)

    return X_test, y_test, labels_test

def convert_class_name_to_int(class_name: str) -> int:
    class_names = {
        'ModerateDemented': 3,
        'MildDemented': 2,
        'VeryMildDemented': 1,
        'NonDemented': 0,
    }
    return class_names[class_name]

def from_score_to_class(class_weights: dict ,score: float, bounds : list[float] = None) -> str:
    # get bounds from class_weights, bounds are the values between the classes
    class_weights_list = list(class_weights.values())
    class_weights_list.sort()
    if bounds == None:
        bounds = []
        for i in range(len(class_weights_list)-1):
            bounds.append((class_weights_list[i]+class_weights_list[i+1])/2)
    #print(bounds)
    # get class from bounds
    for i in range(len(bounds)):
        if score < bounds[i]:
            return i
    return len(bounds)

def convert_labels(labels: np.ndarray) -> np.ndarray:
    labels_converted = []
    for label in labels:
        labels_converted.append(convert_class_name_to_int(label[0]))
    return np.array(labels_converted)

def convert_scores(scores: np.ndarray, class_weights: dict, bounds : list[float] = None) -> np.ndarray:
    scores_converted = []
    for score in scores:
        scores_converted.append(from_score_to_class(class_weights, score[0], bounds))
    return np.array(scores_converted)