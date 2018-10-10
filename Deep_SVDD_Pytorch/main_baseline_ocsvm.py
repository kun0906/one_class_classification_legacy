# -*- coding: utf-8 -*-
"""
    "Using one class classification (ocsvm) to detect abnormal traffic"

    Created at :
        2018/10/04

    Version:
        0.0.1

    Author:

"""
import time

from Utilities.utilies import load_data, dump_model, load_model
from baisc_svm import OCSVM


def ocsvm_main(input_data='csv', kernel='rbf', out_path='./log', **kwargs):
    """

    :param input_data:
    :param kernel:
    :param out_path:
    :param kwargs:
    :return:
    """
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)

    # step 1. load data
    train_set, val_set, test_set = load_data(input_data, norm_flg=True)

    # step 2.1 initialize OC-SVM
    ocsvm = OCSVM(train_set=train_set, kernel=kernel, grid_search_cv_flg=True,val_set='')

    # step 2.2 train OC-SVM model
    ocsvm.train()

    # step 3.1 dump model
    out_file = out_path + "/model.p"
    dump_model(ocsvm, out_file)

    # step 3.2 load model
    model = load_model(input_file=out_file)

    # step 4 evaluate model
    model.evaluate(train_set, name='train_set')
    model.evaluate(test_set, name='test_set')

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))


if __name__ == '__main__':
    dataset = 'mnist'
    dataset = '../data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv'
    ocsvm_main(dataset, kernel='rbf')
