import argparse
import gzip
import os

# from config import Configuration as Cfg
# from utils.log import log_exp_config, log_SVM, log_AD_results
from baisc_svm import OCSVM
from utilies import dump_model, load_model
from utils.visualization.images_plot import plot_outliers_and_most_normal
import numpy as np


def load_data(dataset):
    from Mnist_data_loader import MNIST_DataLoader

    # load data with data loader
    dataset = MNIST_DataLoader(ad_experiment=1)
    train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
        dataset._X_test, dataset._y_test)

    return train_set, val_set, test_set


def ocsvm_main(dataset='mnist', loss='OneClassSVM', kernel='rbf', out_path='./log', **kwargs):
    """

    :param nu: SVM nu
    :param grid_search_CV:
    :return:
    """

    train_set, val_set, test_set = load_data(dataset)
    # initialize OC-SVM
    ocsvm = OCSVM(loss=loss, train_set=train_set, val_set=val_set, kernel=kernel, grid_search_cv_flg=True)

    # train OC-SVM model
    ocsvm.train()

    # predict scores
    ocsvm.evaluate(train_set, name='train_set')

    ocsvm.evaluate(test_set, name='test_set')

    # # log
    # log_exp_config(Cfg.xp_path, args.dataset)
    # log_SVM(Cfg.xp_path, args.loss, args.kernel, ocsvm.gamma, ocsvm.nu)
    # log_AD_results(Cfg.xp_path, ocsvm)
    #
    # # pickle/serialize
    out_file = out_path + "/model.p"
    dump_model(ocsvm, out_file)
    # ocsvm.log_results(filename=out_path + "/AD_results.p")

    # plot targets and outliers sorted
    n_img = 32
    plot_outliers_and_most_normal(ocsvm, n_img, out_path)

    # load model
    model = load_model(input_file=out_file)
    model.evaluate(val_set, name='val_set')


if __name__ == '__main__':
    ocsvm_main(dataset='mnist', loss='OneClassSVM', kernel='rbf')
