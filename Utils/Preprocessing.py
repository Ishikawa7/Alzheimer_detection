import os
import numpy as np
from PIL import Image
import random

def class_occurences(dir_train: str = "Train/" , dir_test: str = "Test/") -> tuple[dict, dict, dict]:
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

    # calculate class weights
    class_weights = dict_train_count.copy()
    class_weights.pop("Total")
    for key in class_weights.keys():
        class_weights[key] = sum_train/dict_train_count[key]
    
    # normalize class weights
    max_weight = max(class_weights.values())
    min_weight = min(class_weights.values())
    for key in class_weights.keys():
        class_weights[key] = (class_weights[key]-min_weight)/(max_weight-min_weight)

    return dict_train_count, dict_test_count, class_weights

def normalize_one_by_one(np_array_image : np.ndarray) -> np.ndarray:
    # normalize with min-max
    np_array_image = np_array_image/np_array_image.max()
    return np_array_image

def shuffle(np_array_list: list[np.ndarray]) -> list[np.ndarray]:
    combined_list = list(zip(np_array_list))
    random.shuffle(combined_list)
    np_array_list = zip(*combined_list)
    return np_array_list

def to_numpy_arrays(class_weights: dict, dir_train :str = "Train/", dir_test :str = "Test/", get_train : bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images_list_train = []
    labels_list_train = []
    class_weights_list_train = []
    images_list_test = []
    labels_list_test = []
    class_weights_list_test = []

    for folder in os.listdir(dir_train):
        for image in os.listdir(dir_train+folder):
            image_file = Image.open(dir_train+folder+"/"+image)
            np_image = np.array(image_file)
            np_image = np_image/np_image.max()
            image_file.close()
            images_list_train.append(np_image)
            labels_list_train.append(folder)
            class_weights_list_train.append(class_weights[folder])

    # shuffle
    combined_list = list(zip(images_list_train, labels_list_train, class_weights_list_train))
    random.shuffle(combined_list)
    images_list_train, labels_list_train, class_weights_list_train = zip(*combined_list)


    for folder in os.listdir(dir_test):
        for image in os.listdir(dir_test+folder):
            image_file = Image.open(dir_test+folder+"/"+image)
            np_image = np.array(image_file)
            np_image = np_image/np_image.max()
            image_file.close()
            images_list_test.append(np_image)
            labels_list_test.append(folder)
            class_weights_list_test.append(class_weights[folder])

    # shuffle
    combined_list = list(zip(images_list_test, labels_list_test, class_weights_list_test))
    random.shuffle(combined_list)
    images_list_test, labels_list_test, class_weights_list_test = zip(*combined_list)

    X_train = np.array(images_list_train)
    y_train = np.array(class_weights_list_train)
    y_train = y_train.reshape(-1,1)
    labels_train = np.array(labels_list_train)
    labels_train = labels_train.reshape(-1,1)
    
    X_test = np.array(images_list_test)
    y_test = np.array(class_weights_list_test)
    y_test = y_test.reshape(-1,1)
    labels_test = np.array(labels_list_test)
    labels_test = labels_test.reshape(-1,1)

    if get_train:
        return X_train, y_train, labels_train
    else:
        return X_test, y_test, labels_test