# -*- coding: utf-8 -*-
"""
    useful tools
"""
import pickle
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from Utilities.CSV_Dataloaer import csv_dataloader


def normalizate_data(np_arr, eplison=10e-4):
    """

    :param np_arr:
    :param eplison: handle with 0.
    :return:
    """
    min_val = np.min(np_arr, axis=0)  # X
    max_val = np.max(np_arr, axis=0)
    range_val = (max_val - min_val)
    if not range_val.all():  # Returns True if all elements evaluate to True.
        for i in range(len(range_val)):
            if range_val[i] == 0.0:
                range_val[i] += eplison
    print('range_val is ', range_val)
    norm_data = (np_arr - min_val) / range_val

    return norm_data


def split_data():
    # train_tset_split()
    pass


def load_data(input_data='', norm_flg=True):
    """

    :param input_data:
    :param normalization_flg:
    :return:
    """
    if 'mnist' in input_data:
        from Utilities.Mnist_data_loader import MNIST_DataLoader
        # load data with data loader
        dataset = MNIST_DataLoader(ad_experiment=1)
        train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
            dataset._X_test, dataset._y_test)
    elif 'csv' in input_data:
        # train_set, val_set, test_set = csv_dataloader(input_data,norm_flg)
        (X, y) = csv_dataloader(input_data)
        if norm_flg:
            X = normalizate_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        train_set, val_set, test_set = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    else:
        print('error dataset.')
        return -1

    # if norm_flg:
    #     train_set = (normalizate_data(train_set[0]),train_set[1]) # X, y
    #     val_set=(normalizate_data(val_set[0]),val_set[1])
    #     test_set=(normalizate_data(test_set[0]),test_set[1])

    return train_set, val_set, test_set


def dump_model(model, out_file):
    """

    :param model:
    :param out_file:
    :return:
    """
    with open(out_file, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved in %s" % out_file)


def load_model(input_file):
    print("Loading model...")

    with open(input_file, 'rb') as f:
        model = pickle.load(f)

    print("Model loaded.")
    return model


def get_variable_name(data_var):
    """
        get variable name as string
    :param data_var:
    :return:
    """
    name = ''
    keys = locals().keys()
    for key, val in locals().items():
        # if id(key) == id(data_var):
        print(key, id(key), id(data_var), key is data_var)
        # if id(key) == id(data_var):
        if val == data_var:
            name = key
            break

    return name
