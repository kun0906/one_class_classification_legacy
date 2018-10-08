# -*- coding: utf-8 -*-

import pickle


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
