# -*- coding: utf-8 -*-
"""
    Purpose:
        Implement one-class classification based on SVDD to detect abnormal data.

"""


from NeuralNet import SVDD_NeuralNet, TrafficDataset


def load_data(input_file=''):
    from Mnist_data_loader import MNIST_DataLoader

    # load data with data loader
    dataset = MNIST_DataLoader()
    train_set, val_set, test_set=dataset.load_data()

    return train_set, val_set, test_set


def SVDD_run(input_file=''):
    """

    :param dataset:
    :param n_epochs:
    :return:
    """

    train_set, val_set, test_set = load_data(input_file)

    # Parameters:
    n_epochs = 2

    # train
    dp_svdd = SVDD_NeuralNet(dataset=TrafficDataset(train_set),n_epochs=n_epochs, use_weights='', pretrain=0)
    dp_svdd.train(save_at='./data', save_to='./results/')
    #
    # # pickle/serialize AD results
    # if Cfg.ad_experiment:
    #     nnet.log_results(filename=Cfg.xp_path + "/AD_results.p")
    #
    # # plot diagnostics
    # if Cfg.nnet_diagnostics:
    #     # common suffix for plot titles
    #     str_lr = "lr = " + str(args.lr)
    #     C = int(args.C)
    #     if not Cfg.weight_decay:
    #         C = None
    #     str_C = "C = " + str(C)
    #     Cfg.title_suffix = "(" + args.solver + ", " + str_C + ", " + str_lr + ")"
    #
    #     if args.loss == 'autoencoder':
    #         plot_ae_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)
    #     else:
    #         plot_diagnostics(nnet, Cfg.xp_path, Cfg.title_suffix)
    #
    # plot_filters(nnet, Cfg.xp_path, Cfg.title_suffix)
    #
    # # If AD experiment, plot most anomalous and most normal
    # if Cfg.ad_experiment:
    #     n_img = 32
    #     plot_outliers_and_most_normal(nnet, n_img, Cfg.xp_path)


if __name__ == '__main__':
    input_file = ''
    SVDD_run(input_file)
