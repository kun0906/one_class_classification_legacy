# -*- coding: utf-8 -*-
"""
    implement one class classification (OCSVM)

     Created at :
        2018/10/04

    Version:
        0.0.1

    Author:
"""

import time
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


class OCSVM(object):

    def __init__(self, train_set, kernel='rbf', grid_search_cv_flg=False, val_set='', **kwargs):
        """

        :param train_set:
        :param kernel:
        :param grid_search_cv_flg:
        :param val_set:
        :param kwargs:
        """
        # init
        self.train_set_with_labels = train_set  # used for evaluation
        X = np.asarray([x_t for (x_t, y_t) in zip(*train_set) if y_t == 0], dtype=float)
        print('X.shape: ', X.shape)

        self.train_set = X  # only X
        self.kernel = kernel
        self.nu = 0.5
        # self.gamma=1
        self.grid_search_cv_flg = grid_search_cv_flg
        self.val_set = val_set
        if self.grid_search_cv_flg:
            if len(val_set) == 0:
                print('re-split train_set into train_set and val_set.')
                X, y = self.train_set_with_labels[0], self.train_set_with_labels[1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
                self.train_set_with_labels = (X_train, y_train)
                self.val_set = (X_test, y_test)
            # assert len(val_set) == 0

        # Scores and AUC
        self.diag = {}

        self.diag['train_set'] = {}
        self.diag['val_set'] = {}
        self.diag['test_set'] = {}

        self.diag['train_set']['scores'] = {}
        self.diag['val_set']['scores'] = {}
        # self.diag['test_set']['scores'] = np.zeros((len(self.data._y_test), 1))
        self.diag['test_set']['scores'] = {}

        self.diag['train_set']['auc'] = np.zeros(1)
        self.diag['val_set']['auc'] = np.zeros(1)
        self.diag['test_set']['auc'] = np.zeros(1)

        self.diag['train_set']['acc'] = np.zeros(1)
        self.diag['val_set']['acc'] = np.zeros(1)
        self.diag['test_set']['acc'] = np.zeros(1)

        # # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def train(self):
        print('Training begins...')
        start_time = time.time()
        # X_train_shape = self.train_set[0].shape
        # X_train = self.train_set[0].reshape(X_train_shape[0], np.prod(X_train_shape[1:]))
        X_train = self.train_set
        print('\tTrain_set size is ', X_train.shape)

        if self.grid_search_cv_flg:
            # parameters={'kernel':('linear','rbf'),'nu':[0.01,0.99],'gamma':[0,1]}
            # self.ocsvm=svm.OneClassSVM()
            # clf = GridSearchCV(self.ocsvm,parameters,cv=5,scoring="accuracy")
            # clf.fit(X_train)
            cv_auc = 0.0
            cv_acc = 0.0
            for nu in np.logspace(-10, -0.001, num=3, base=2):
                for gamma in np.logspace(-10, -0.001, num=3, base=2):  # log2
                    # train on selected gamma
                    print('nu:', nu, ', gamma:', gamma)
                    self.ocsvm = svm.OneClassSVM(kernel=self.kernel, nu=nu, gamma=gamma)
                    self.ocsvm.fit(X_train)
                    # predict on small hold-out set
                    auc, acc, cm = self.evaluate(self.val_set, name='val_set')
                    # save model if AUC on hold-out set improved
                    # if self.diag['val_set']['auc'][0] > cv_auc:
                    if self.diag['val_set']['acc'][0] > cv_acc:
                        self.best_ocsvm = self.ocsvm  # save the best results
                        self.nu = nu
                        self.gamma = gamma
                        cv_auc = self.diag['val_set']['auc'][0]
                        cv_acc = self.diag['val_set']['acc'][0]
                        self.auc = auc
                        self.acc = acc
                        self.cm = cm
            # save results of best cv run
            self.diag['val_set']['auc'] = cv_auc
            self.diag['val_set']['acc'] = cv_acc

            print('---The best accuracy on \'val_set\' is %.2f%% when nu and gamma are %.5f and %.5f respectively' % (
                self.acc, self.nu, self.gamma))
            print('---Confusion matrix:\n', self.cm)
            self.ocsvm = self.best_ocsvm
        else:
            # if rbf-kernel, re-initialize svm with gamma minimizing the numerical error
            gamma = 1 / (np.max(pairwise_distances(X_train)) ** 2)
            print('gamma:', gamma)
            # gamma = 0.7
            self.ocsvm = svm.OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=gamma)  # construction function

            self.ocsvm.fit(X_train)
        print('Training finished, it takes %.2fs' % (time.time() - start_time))

    def evaluate(self, data_set, name='test_set', **kwargs):
        """

        :param data_set:
        :param name:
        :param kwargs:
        :return:
        """

        start_time = time.time()
        # name = self.get_variable_name(data_set)
        # name = [k for k,v in locals().items() if v == data_set]
        print('\t Evaluating data is \'%s\'.' % name)

        self.diag[name]['scores'] = np.zeros((len(data_set[1]), 1), dtype=float)

        X = data_set[0]
        y = data_set[1]
        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

        print("Evaluation begins...")
        scores = (-1.0) * self.ocsvm.decision_function(X)
        y_pred = (self.ocsvm.predict(X) == -1) * 1

        cm = confusion_matrix(y, y_pred)
        print(name + ' Confusion matrix:\n', cm)
        acc = 100.0 * sum(y == y_pred) / len(y)
        print(name + ' Acc: %.2f%% ' % (acc))

        self.diag[name]['scores'][:, 0] = scores.flatten()
        print(100.0 * sum(y == y_pred) / len(y))
        self.diag[name]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)

        if sum(y) > 0:
            auc = roc_auc_score(y, scores.flatten())
            self.diag[name]['auc'][0] = auc

        print('Evaluation finished, it takes %.2fs' % (time.time() - start_time))

        return auc, acc, cm
