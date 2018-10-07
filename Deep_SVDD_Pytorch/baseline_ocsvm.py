import argparse
import gzip
import os

# from config import Configuration as Cfg
from svm import OCSVM
# from utils.log import log_exp_config, log_SVM, log_AD_results
from utils.visualization.images_plot import plot_outliers_and_most_normal
import numpy as np

def load_data(dataset):
    from Mnist_data_loader import MNIST_DataLoader

    # load data with data loader
    # dataset = MNIST_DataLoader()
    with gzip.open('/home/kun/PycharmProjects/Deep_SVDD_20181006/Deep-SVDD-master/data/train-images-idx3-ubyte.gz', 'rb') as f:
            train_set = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)

    train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
    dataset._X_test, dataset._y_test)

    return train_set, val_set, test_set


def ocsvm_main(dataset='mnist', loss='OneClassSVM', kernel='rbf'):
    """

    :param nu: SVM nu
    :param grid_search_CV:
    :return:
    """

    train_set, val_set, test_set = load_data(dataset)
    # initialize OC-SVM
    ocsvm = OCSVM(loss=loss, train_set=train_set, val_set=val_set, kernel=kernel, GridSearch=True)

    # train OC-SVM model
    ocsvm.train()

    # predict scores
    ocsvm.predict(train_set)
    ocsvm.predict(test_set)

    # # log
    # log_exp_config(Cfg.xp_path, args.dataset)
    # log_SVM(Cfg.xp_path, args.loss, args.kernel, ocsvm.gamma, ocsvm.nu)
    # log_AD_results(Cfg.xp_path, ocsvm)
    #
    # # pickle/serialize
    # ocsvm.dump_model(filename=Cfg.xp_path + "/model.p")
    # ocsvm.log_results(filename=Cfg.xp_path + "/AD_results.p")

    # plot targets and outliers sorted
    n_img = 32
    plot_outliers_and_most_normal(ocsvm, n_img, Cfg.xp_path)


if __name__ == '__main__':
    ocsvm_main(dataset='mnist', loss='OneClassSVM', kernel='rbf')
