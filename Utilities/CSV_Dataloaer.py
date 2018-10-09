# -*- coding: utf-8 -*-
"""
    load data from csv file
"""
from collections import Counter
import numpy as np


def csv_dataloader(input_file):
    """

    :param input_file:
    :return:
    """
    X = []
    y = []
    with open(input_file, 'r') as f_in:
        line = f_in.readline()
        while line:
            if line.startswith('Flow'):
                line = f_in.readline()
            line_arr = line.split(',')
            X.append(line_arr[7:40])
            if line_arr[-1] == '2\n':
                y.append('1')
            else:
                y.append('0')

            line = f_in.readline()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    print(Counter(y))

    return (X, y)
