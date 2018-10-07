"""

    Created at: 20181006
"""
import time
import numpy as np
from torch.utils.data import Dataset

__author__='kun'

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
# import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F


def print_network(describe_str, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


class SVDD_NeuralNet(nn.Module):

    def __init__(self, *args, **kwargs):
        # super(ANN,self).__init__() # python 2.x
        super().__init__()  # python 3.x

        self.dataset=kwargs['dataset']
        self.n_epochs = kwargs['n_epochs']
        self.batch_size= 64
        # self.batch_size = kwargs['BATCH_SIZE']
        # self.first_n_pkts = kwargs['first_n_pkts']
        # self.out_size = kwargs['num_class']

        first_n_pkts = 1
        self.small_in_size = first_n_pkts
        self.small_h_size = 5
        self.small_out_size = 2

        # self.pkts_ann = nn.Sequential(nn.Linear(self.small_in_size, self.small_h_size * 2), nn.Tanh(),
        #                               nn.Linear(self.small_h_size * 2, self.small_h_size), nn.Tanh(),
        #                               nn.Linear(self.small_h_size, self.small_out_size)
        #                               )
        #
        # self.intr_tm_ann = nn.Sequential(nn.Linear(self.small_in_size, self.small_h_size * 2), nn.Tanh(),
        #                                  nn.Linear(self.small_h_size * 2, self.small_h_size), nn.Tanh(),
        #                                  nn.Linear(self.small_h_size, self.small_out_size)
        #                                  )

        # self.in_size = 2 * self.small_out_size + 1  # first_n_pkts_list, flow_duration, intr_time_list
        self.in_size = 60
        self.h_size = 5
        # self.out_size = 1  # number of label, one-hot coding
        # self.classify_ann = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
        #                                   nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
        #                                   nn.Linear(self.h_size, self.out_size, nn.Softmax())
        #                                   )

        # For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
        #
        # If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

        # self.conv1 = nn.Conv2d(1, 6, (5, 1), stride=1)
        # self.conv2 = nn.Conv2d(6, 16, (5, 1), stride=1)
        # # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 1 * 5, 120)
        self.fc1 = nn.Linear(784, 120,bias=False) # params=(in,out,bias=True)
        self.fc2 = nn.Linear(120, 84,bias=False)
        self.fc3 = nn.Linear(84, 1,bias=False)

        # self.classify_ann = nn.Sequential(self.conv1, nn.Tanh(),
        #                                   self.conv2, nn.Tanh(),
        #                                   self.fc1,nn.Tanh(),
        #                                   self.fc2,nn.Tanh(),
        #                                   self.fc3
        #                                   )

        print('---------- Networks architecture -------------')
        # print_network('pkts_ann:', self.pkts_ann)
        # print_network('intr_tm_ann:', self.intr_tm_ann)
        # print_network('classify_ann:', self.classify_ann)
        # print('-----------------------------------------------')

        self.criterion = nn.MSELoss(reduction='sum')
        # self.criterion = nn.MultiLabelMarginLoss()
        self.learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam([self.pkts_ann, self.intr_tm_ann, self.classify_ann], lr=self.d_learning_rate,
        #                             betas=(0.5, 0.9))
        # params = list(self.pkts_ann.parameters()) + list(self.intr_tm_ann.parameters()) + list(
        #     self.classify_ann.parameters())

        self.R_val=nn.parameter.Parameter(torch.randn(1))
        params = list(self.parameters())
        params.append(self.R_val)
        self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=(0.5, 0.9))

    def svdd_loss(self, net_outs):
        """

        :param net_outs:
        :return:
        """
        c=0.0
        self.lambda_v=0.8
        loss_func=nn.MSELoss()
        loss = loss_func(net_outs,(torch.ones(net_outs.shape[0],1)*self.R_val).view(net_outs.shape[0],1))
        ##+ self.lambda_v * self.parameters()

        return loss

        # objective = 0.0
        # accuracy = 0.0

        # # Backpropagation (without training R)
        # obj = T.cast(floatX(0.5) * (l2_penalty + rec_penalty) + nnet.Rvar + loss,
        #              dtype='floatX')
        # updates = get_updates(nnet, obj, trainable_params, solver=nnet.solver)
        # nnet.backprop_without_R = theano.function([inputs, targets], [obj, acc], updates=updates,
        #                                           on_unused_input='warn')
        #
        # err, acc, b_scores, l2, b_rec, b_rep, b_rep_norm, _, b_loss, R = self.forward(inputs, targets)
        #
        # scores[start_idx:stop_idx] = b_scores.flatten()
        # rep[start_idx:stop_idx, :] = b_rep
        # rep_norm[start_idx:stop_idx] = b_rep_norm
        # reconstruction_penalty += b_rec
        #
        #
        #
        #     objective += err
        #     accuracy += acc
        #     emp_loss += b_loss
        #     batches += 1
        #
        # objective /= batches
        # accuracy *= 100. / batches
        # emp_loss /= batches
        # reconstruction_penalty /= batches

        # if print_:
        #     print_obj_and_acc(objective, accuracy, which_set)

        # # save diagnostics
        #
        # nnet.save_objective_and_accuracy(epoch, which_set, objective, accuracy)
        # nnet.save_diagnostics(which_set, epoch, scores, rep_norm, rep, emp_loss, reconstruction_penalty)
        #
        # # Save network parameter diagnostics (only once per epoch)
        # if which_set == 'train':
        #     nnet.save_network_diagnostics(epoch, floatX(l2), floatX(R))
        #
        # # Track results of epoch with highest AUC on test set
        # if which_set == 'test' and (nnet.data.n_classes == 2):
        #     nnet.track_best_results(epoch)

        # return objective, accuracy

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_c_as_mean(self,params=''):
        pass

    def train(self, save_at, save_to):
        """

        :param save_at:
        :param save_to:
        :return:
        """

        print("Starting training with %s" % self.optimizer)

        # initialize c from mean of network feature representations in deep SVDD if specified
        if True:
            self.initialize_c_as_mean(self.optimizer.param_groups)

        # train on epoch
        train_loader = Data.DataLoader(
            dataset=self.dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=2,
        )
        for epoch in range(self.n_epochs):

            # # get copy of current network parameters to track differences between epochs
            # nnet.copy_parameters()

            # In each epoch, we do a full pass over the training data:
            start_time = time.time()

            for step, (b_x, b_y) in enumerate(
                    train_loader):  # type: (int, (object, object)) # gives batch data, normalize x when iterate train_loader
                net_outs = self.forward(b_x.view(b_x.shape[0],-1).float())
                # loss = self.criterion(y_preds, b_y.view(-1,1).float())  # net_outs, y_real(targets)
                loss = self.svdd_loss(net_outs)

                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                # self.train_hist['loss'].append(loss.data.tolist())
                # self.test_hist['loss'].append(loss.data.tolist())

                if step % 100 == 0:
                    print('epoch = %d, loss = %f' % (epoch, loss.data.tolist()))


class TrafficDataset(Dataset):

    def __init__(self,input_data, transform=None, normalization_flg=False):
        self.X = input_data[0]
        self.y = input_data[1]
        # with open(input_file, 'r') as fid_in:
        #     line = fid_in.readline()
        #     while line:
        #         line_arr = line.split(',')
        #         value = list(map(lambda x: float(x), line_arr[:-1]))
        #         self.X.append(value)
        #         self.y.append(float(line_arr[-1].strip()))
        #         line = fid_in.readline()
        #
        # # if normalization_flg:
        # #     self.X = normalize_data(np.asarray(self.X, dtype=float), range_value=[-1, 1], eps=1e-5)

        self.transform = transform

    def __getitem__(self, index):

        value_x = self.X[index]
        value_y = self.y[index]
        if self.transform:
            value_x = self.transform(value_x)

        value_x = torch.from_numpy(np.asarray(value_x)).double()
        value_y = torch.from_numpy(np.asarray(value_y)).double()

        # X_train, X_test, y_train, y_test = train_test_split(value_x, value_y, train_size=0.7, shuffle=True)
        return value_x, value_y  # Dataset.__getitem__() should return a single sample and label, not the whole dataset.
        # return value_x.view([-1,1,-1,1]), value_y

    def __len__(self):
        return len(self.X)
