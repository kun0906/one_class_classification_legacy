# -*- coding: utf-8 -*-
"""
   @ autoencoder:
           abnormal detection

"""
from sklearn.metrics import confusion_matrix
from Utilities.utilies import load_data

__author__ = 'Learn-Live'


import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader


def print_net(net, describe_str='Net'):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


class AutoEncoder(nn.Module):
    def __init__(self, X, y, epochs=2):
        """

        :param X: Features
        :param y: Labels
        :param epochs:
        """
        super().__init__()
        # self.dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
        self.dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(X))

        self.epochs = epochs
        self.learning_rate = 1e-3
        self.batch_size = 50

        self.show_flg = True

        self.num_features_in = len(X[0])
        self.h_size = 16
        self.num_latent_features = 10
        self.num_features_out = self.num_features_in

        self.encoder = nn.Sequential(
            nn.Linear(self.num_features_in, self.h_size * 8),
            nn.ReLU(True),
            nn.Linear(self.h_size * 8, self.h_size * 4),
            nn.ReLU(True),
            nn.Linear(self.h_size * 4, self.h_size * 2),
            nn.ReLU(True),
            nn.Linear(self.h_size * 2, self.num_latent_features))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_latent_features, self.h_size * 2),
            nn.ReLU(True),
            nn.Linear(self.h_size * 2, self.h_size * 4),
            nn.ReLU(True),
            nn.Linear(self.h_size * 4, self.h_size * 8),
            nn.ReLU(True),
            nn.Linear(self.h_size * 8, self.num_features_in),
            nn.Sigmoid())

        if self.show_flg:
            print_net(self.encoder, describe_str='Encoder')
            print_net(self.decoder, describe_str='Decoder')

        self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.loss = []
        for epoch in range(self.epochs):
            for iter, (batch_X, _) in enumerate(dataloader):
                # # ===================forward=====================
                output = self.forward(batch_X)
                loss = self.criterion(output, batch_X)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss.append(loss.data)

            # ===================log========================
            print('epoch [{:d}/{:d}], loss:{:.4f}'
                  .format(epoch + 1, self.epochs, loss.data))
            # if epoch % 10 == 0:
            #     # pic = to_img(output.cpu().data)
            #     # save_image(pic, './mlp_img/image_{}.png'.format(epoch))

        self.T = self.loss[-1]
        if self.show_flg:
            plt.figure()

            plt.plot(self.loss, 'r', alpha=0.5, label='loss')
            # plt.plot(G_loss, 'g', alpha=0.5, label='G_loss')
            plt.legend(loc='upper right')
            plt.show()

    def evaluate(self, test_set):
        """

        :param test_set:
        :return:
        """
        X = torch.Tensor(test_set[0])
        y_true = test_set[1]

        ### predict output
        AE_outs = self.forward(X)

        y_preds = []
        num_abnormal = 0
        print('Threshold(T) is ', self.T.data.tolist())
        for i in range(X.shape[0]):
            if torch.norm((AE_outs[i] - X[i]), 2) > self.T:
                # print('abnormal sample.')
                y_preds.append('1')  # 0 is normal, 1 is abnormal
                num_abnormal += 1
            else:
                y_preds.append('0')
        # if torch.dist(AE_outs, X, 2) > self.T:
        #     print('abnormal sample.')
        #     y_preds.append('1')  # 0 is normal, 1 is abnormal
        #     num_abnormal += 1
        # else:
        #     y_preds.append('0')
        y_preds = np.asarray(y_preds, dtype=int)
        cm = confusion_matrix(y_pred=y_preds, y_true=y_true)
        print('Confusion matrix:\n', cm)
        acc = 100.0 * sum(y_true == y_preds) / len(y_true)
        print('Acc: %.2f%%' % acc)


def main(input_file, epochs=2):
    """

    :param input_file: CSV
    :return:
    """
    torch.manual_seed(1)
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)

    ### 1. load data and do preprocessing
    train_set, val_set, test_set = load_data(input_file,norm_flg=True)
    X= np.asarray([x_t for (x_t, y_t) in zip(*train_set) if y_t == 0],dtype=float)
    # X = []
    # for (x_t, y_t) in zip(*train_set):
    #     if y_t == 0:
    #         X.append(x_t)
    #         # y.append(y_t)
    #
    # X= np.asarray(X,dtype=float)
    print('X.shape: ',X.shape)

    ### 2. model initialization
    AE_model = AutoEncoder(X=X, y='', epochs=epochs)
    ### a. train model
    AE_model.train()

    ### b. dump model
    model_path = './log/autoencoder.pth'
    torch.save(AE_model, model_path)

    ### c. load model
    AE_model = torch.load(model_path)

    ### d. evaluate model
    AE_model.evaluate(train_set)
    AE_model.evaluate(test_set)

    ###
    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))


if __name__ == '__main__':
    input_file = '../data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv'
    epochs = 10
    main(input_file, epochs)
