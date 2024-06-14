# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import enable_grad
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import timeit

import logging

"""
define net
##############################################################################
"""


class MLP(nn.Module):
    def __init__(
        self,
        i_dim=14,
        h_dim=[30, 15],
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
        use_bias=True,
    ):
        super().__init__()
        self.use_bias = use_bias
        # init module list
        self.module_list = nn.ModuleList()

        # get hidden layer's list
        # wrap h_dim in list of it's not already
        if not isinstance(h_dim, list):
            try:
                h_dim = [h_dim]
            except Exception as e:
                logging.error(e)
        # add i_dim to h_dim
        h_dim.insert(0, i_dim)

        # get if bias should be used or not
        for k in range(len(h_dim) - 1):
            # add linear layer
            self.module_list.append(
                nn.Linear(h_dim[k], h_dim[k + 1], bias=self.use_bias)
            )
            # add nonlinearity
            if nlin == "elu":
                self.module_list.append(nn.ELU())
            if nlin == "celu":
                self.module_list.append(nn.CELU())
            if nlin == "gelu":
                self.module_list.append(nn.GELU())
            if nlin == "leakyrelu":
                self.module_list.append(nn.LeakyReLU())
            if nlin == "relu":
                self.module_list.append(nn.ReLU())
            if nlin == "tanh":
                self.module_list.append(nn.Tanh())
            if nlin == "sigmoid":
                self.module_list.append(nn.Sigmoid())
            if nlin == "silu":
                self.module_list.append(nn.SiLU())
            if dropout > 0:
                self.module_list.append(nn.Dropout(dropout))
        # init output layer
        self.module_list.append(nn.Linear(h_dim[-1], o_dim, bias=self.use_bias))
        # normalize outputs between 0 and 1
        # self.module_list.append(nn.Sigmoid())

        # initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                if self.use_bias:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            logging.debug(f"layer {layer}")
            logging.debug(f"input shape:: {x.shape}")
            x = layer(x)
            logging.debug(f"output shape:: {x.shape}")
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            activations.append(x)
        return x, activations


###############################################################################
# define net
# ##############################################################################
def compute_outdim(i_dim, stride, kernel, padding, dilation):
    o_dim = (i_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return o_dim


class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class CNN2(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


################################################################################################
class ResCNN(nn.Module):
    """
    extension of the CNN class.
    Added residual connections via max-pooling and 1d convs to the conv layers
    Added a function to load state-dicts from the cnns without res cons
    """

    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError(
                "dropout is not yet impemented for the residual connections.|"
            )
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        # input shape bx1x28x28

        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 8, 12, 12]
        ## residual connection stack 1
        self.res1_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res1_conv = nn.Conv2d(
            in_channels=channels_in, out_channels=8, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 6, 4, 4]
        self.res2_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res2_conv = nn.Conv2d(
            in_channels=8, out_channels=6, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## output [15, 4, 3, 3]
        self.res3_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.res3_conv = nn.Conv2d(
            in_channels=6, out_channels=4, kernel_size=1, stride=1, padding=0
        )

        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        for m in self.module_list:
            m = self.init_single(init_type, m)

        init_type = "kaiming_uniform"  # init the residual blocks with kaiming always
        self.res1_conv = self.init_single(init_type, self.res1_conv)
        self.res2_conv = self.init_single(init_type, self.res2_conv)
        self.res3_conv = self.init_single(init_type, self.res3_conv)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            m.bias.data.fill_(0.01)
        return m

    def get_nonlin(self, nlin):
        """
        gets nn class object for keyword
        """
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def load_weights_from_cnn_checkpoint(self, check):
        """
        takes weights and biases from CNN architecture without residual connections
        assumes this particular data structure, will not fail gracefully otherwise
        """
        self.module_list[0].weight.data = check["module_list.0.weight"]
        self.module_list[0].bias.data = check["module_list.0.bias"]
        self.module_list[3].weight.data = check["module_list.3.weight"]
        self.module_list[3].bias.data = check["module_list.3.bias"]
        self.module_list[6].weight.data = check["module_list.6.weight"]
        self.module_list[6].bias.data = check["module_list.6.bias"]
        self.module_list[9].weight.data = check["module_list.9.weight"]
        self.module_list[9].bias.data = check["module_list.9.bias"]
        self.module_list[11].weight.data = check["module_list.11.weight"]
        self.module_list[11].bias.data = check["module_list.11.bias"]

    def forward(self, x):
        # layer 1
        x_ = x.clone()
        for idx in range(0, 3):
            x_ = self.module_list[idx](x_)
        x = self.res1_pool(x)
        x = self.res1_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 2
        x_ = x.clone()
        for idx in range(3, 6):
            x_ = self.module_list[idx](x_)
        x = self.res2_pool(x)
        x = self.res2_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 3
        x_ = x.clone()
        for idx in range(6, 8):
            x_ = self.module_list[idx](x_)
        x = self.res3_pool(x)
        x = self.res3_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # flatten
        # fc1
        # fc2
        for m in self.module_list[8:]:
            x = m(x)

        return x


################################################################################
class CNN_more_layers(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, 8, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 4, 5, bias=False),
            nn.Tanh(),
            nn.Conv2d(4, 6, 4, bias=False),
            nn.Tanh(),
            nn.Conv2d(6, 4, 3, bias=False),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 10),
        )

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        for m in self.layers:
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        return self.layers(x)


###############################################################################
class CNN_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(6, 4, 2, bias=False)
        self.act3 = nn.Tanh()

        self.identity = nn.Conv2d(8, 4, 1, bias=False)

        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(36, 20, bias=False)
        self.act4 = nn.Tanh()

        self.fc5 = nn.Linear(20, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        self.conv1 = self.init_single(init_type, self.conv1)
        self.conv2 = self.init_single(init_type, self.conv2)
        self.conv3 = self.init_single(init_type, self.conv3)
        self.fc4 = self.init_single(init_type, self.fc4)
        self.fc5 = self.init_single(init_type, self.fc5)

        init_type = "kaiming_uniform"
        self.identity = self.init_single(init_type, self.identity)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x_identity = self.identity(x)
        logging.debug("x_identity", x_identity.shape)

        x = self.act2(self.pool2(self.conv2(x)))
        logging.debug(x.shape)

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, ::4, ::4]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act4(self.fc4(x))
        logging.debug(x.shape)

        x = self.fc5(x)
        logging.debug(x.shape)

        return x


###############################################################################
class CNN_more_layers_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(4, 6, 4, bias=False)
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(6, 4, 3, bias=False)
        self.act4 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc5 = nn.Linear(36, 18)
        self.act5 = nn.Tanh()

        self.fc6 = nn.Linear(18, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        self.conv1 = self.init_single(init_type, self.conv1)
        self.conv2 = self.init_single(init_type, self.conv2)
        self.conv3 = self.init_single(init_type, self.conv3)
        self.conv4 = self.init_single(init_type, self.conv4)
        self.fc5 = self.init_single(init_type, self.fc5)
        self.fc6 = self.init_single(init_type, self.fc6)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass

        return m

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x = self.act2(self.conv2(x))
        logging.debug(x.shape)

        x_identity = x.clone()

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = self.act4(self.conv4(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, 1:-1, 1:-1][:, :, ::2, ::2]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act5(self.fc5(x))
        logging.debug(x.shape)

        x = self.fc6(x)
        logging.debug(x.shape)

        return x


import torchvision

from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as ResNetBase


class ResNet(ResNetBase):
    """
    Wrapper for pytroch ResNet class to get access to forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_activations(self, x):
        """forward pass and return activations"""
        # See note [TorchScript super()]
        activations = []

        # input block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        activations.append(x.clone())

        # residual blocks
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())

        # output block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, activations


class ResNetBase(ResNet):
    """
    ResNet base class, defaults to ResNet 18, implements init and weight init
    """

    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(block=block, layers=layers, num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m


class ResNet18(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet34(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet50(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet101(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet152(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 8, 36, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=Bottleneck,
            layers=layers,
        )


class MiniAlexNet(nn.Module):
    def __init__(self, channels_in=3, num_classes=10, init_type="kaiming_uniform"):
        super(MiniAlexNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            channels_in, 96, kernel_size=3, stride=1
        )  # Use padding to keep size 32x32
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=3
        )  # Reduce size from 32x32 to 10x10
        self.batchnorm1 = nn.BatchNorm2d(96)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)  # Keep size 10x10
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Reduce size from 10x10 to 5x5
        self.batchnorm2 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 384)  # Adjusted for 5x5 feature map size
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        # Apply Convolutional Layers
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.maxpool2(x)

        x = x.view(-1, 256 * 4 * 4)

        # Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


###############################################################################
# define FNNmodule
# ##############################################################################
class NNmodule(nn.Module):
    def __init__(self, config, cuda=False, seed=42, verbosity=0):
        super(NNmodule, self).__init__()

        # set verbosity
        self.verbosity = verbosity
        if not cuda:
            cuda = True if config.get("device", "cpu") == "cuda" else False
        if cuda and torch.cuda.is_available():
            self.device = "cuda"
            logging.info("cuda availabe:: use cuda")
        else:
            self.device = "cpu"
            self.cuda = False
            logging.info("cuda unavailable:: fallback to cpu")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # construct model
        if config["model::type"] == "MLP":
            # calling MLP constructor
            logging.info("=> creating model MLP")
            i_dim = config["model::i_dim"]
            h_dim = config["model::h_dim"]
            o_dim = config["model::o_dim"]
            nlin = config["model::nlin"]
            dropout = config["model::dropout"]
            init_type = config["model::init_type"]
            use_bias = config["model::use_bias"]
            model = MLP(i_dim, h_dim, o_dim, nlin, dropout, init_type, use_bias)

        elif config["model::type"] == "CNN":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = CNN(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN2":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = CNN2(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN3":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = CNN3(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "ResCNN":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = ResCNN(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN_more_layers":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = CNN_more_layers(
                init_type=config["model::init_type"],
                channels_in=config["model::channels_in"],
            )
        elif config["model::type"] == "CNN_residual":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = CNN_residual(
                init_type=config["model::init_type"],
                channels_in=config["model::channels_in"],
            )
        elif config["model::type"] == "CNN_more_layers_residual":
            # calling MLP constructor
            logging.info("=> creating model CNN")
            model = CNN_more_layers_residual(
                init_type=config["model::init_type"],
                channels_in=config["model::channels_in"],
            )
        elif config["model::type"] == "Resnet18":
            # calling MLP constructor
            logging.info("=> creating Resnet18")
            model = ResNet18(
                channels_in=config["model::channels_in"],
                out_dim=config["model::o_dim"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "Resnet34":
            # calling MLP constructor
            logging.info("=> creating Resnet34")
            model = ResNet34(
                channels_in=config["model::channels_in"],
                out_dim=config["model::o_dim"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "Resnet50":
            logging.info("=> create resnet50")
            model = ResNet50(
                channels_in=config["model::channels_in"],
                out_dim=config["model::o_dim"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "Resnet101":
            logging.info("=> create resnet101")
            model = ResNet101(
                channels_in=config["model::channels_in"],
                out_dim=config["model::o_dim"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "Resnet152":
            logging.info("=> create resnet152")
            model = ResNet152(
                channels_in=config["model::channels_in"],
                out_dim=config["model::o_dim"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "MiniAlexNet":
            logging.info("=> create MiniAlexNet")
            model = MiniAlexNet(
                channels_in=config["model::channels_in"],
                num_classes=config["model::o_dim"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "efficientnet_v2_s":
            logging.info("=> create efficientnet_v2_s")
            model = torchvision.models.efficientnet_v2_s(
                num_classes=config["model::o_dim"],
                dropout=config["model::dropout"],
            )
        elif config["model::type"] == "efficientnet_v2_m":
            logging.info("=> create efficientnet_v2_m")
            model = torchvision.models.efficientnet_v2_m(
                num_classes=config["model::o_dim"],
                dropout=config["model::dropout"],
            )
        elif config["model::type"] == "densenet121":
            logging.info("=> create densenet121")
            model = torchvision.models.densenet121(
                num_classes=config["model::o_dim"],
            )
        elif config["model::type"] == "densenet161":
            logging.info("=> create densenet161")
            model = torchvision.models.densenet161(
                num_classes=config["model::o_dim"],
            )
        elif config["model::type"] == "vit_b_16":
            logging.info("=> create vit_b_16")
            model = torchvision.models.vit_b_16(
                num_classes=config["model::o_dim"],
                dropout=config["model::dropout"],
            )
        else:
            raise NotImplementedError("error: model type unkown")

        logging.info(f"send model to {self.device}")
        model.to(self.device)

        self.model = model

        # define loss function (criterion) and optimizer
        # set loss
        self.task = config.get("training::task", "classification")
        self.logsoftmax = False
        if self.task == "classification":
            if config.get("training::loss", "nll"):
                self.criterion = nn.NLLLoss()
                self.logsoftmax = True
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif self.task == "regression":
            self.criterion = nn.MSELoss(reduction="mean")
        if self.device == "cuda":
            self.criterion.to(self.device)

        # set opimizer
        self.set_optimizer(config)

        self.set_scheduler(config)

        self.best_epoch = None
        self.loss_best = None

    # module forward function
    def forward(self, x):
        # compute model prediction
        y = self.model(x)
        if self.logsoftmax:
            y = F.log_softmax(y, dim=1)
        return y

    # set optimizer function - maybe we'll only use one of them anyways..
    def set_optimizer(self, config):
        if config["optim::optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config["optim::lr"],
                momentum=config["optim::momentum"],
                weight_decay=config["optim::wd"],
                nesterov=config.get("optim::nesterov", False),
            )
        if config["optim::optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
            )
        if config["optim::optimizer"] == "rms_prop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
                momentum=config["optim::momentum"],
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "OneCycleLR":
            logging.info("use onecycleLR scheduler")
            max_lr = config["optim::lr"]
            steps_per_epoch = config["scheduler::steps_per_epoch"]
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                epochs=config["training::epochs_train"],
                steps_per_epoch=steps_per_epoch,
            )

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    def compile_model(self):
        logging.info("compiling the model")
        self.model = torch.compile(self.model)  # requires PyTorch 2.0
        logging.info("compiled successfully")

    def compute_mean_loss(self, dataloader):
        # step 1: compute data mean
        # get output data
        logging.debug(f"len(dataloader): {len(dataloader)}")

        # get shape of data
        for idx, (_, target) in enumerate(dataloader):
            # unsqueeze scalar targets for compatibility
            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1)
            x_mean = torch.zeros(target.shape[1])
            break

        logging.debug(f"x_mean.shape: {x_mean.shape}")
        n_data = 0
        # collect mean
        for idx, (_, target) in enumerate(dataloader):
            # unsqueeze scalar targets for compatibility
            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1)
            # compute mean weighted with batch size
            n_data += target.shape[0]
            x_mean += target.mean(dim=0) * target.shape[0]

        # scale x_mean back
        x_mean /= n_data
        logging.debug(f"x_mean = {x_mean}")
        n_data = 0
        loss_mean = 0
        # collect loss
        for idx, (_, target) in enumerate(dataloader):
            # unsqueeze scalar targets for compatibility
            if len(target.shape) == 1:
                target = target.unsqueeze(dim=1)
            # compute mean weighted with batch size
            n_data += target.shape[0]
            # broadcast x_mean to target shape
            target_mean = torch.zeros(target.shape).add(x_mean)
            # commpute loss
            loss_batch = self.criterion(target, target_mean)
            # add and weight
            loss_mean += loss_batch.item() * target.shape[0]
        # scale back
        loss_mean /= n_data

        # compute mean
        self.loss_mean = loss_mean
        logging.debug(f" mean loss: {self.loss_mean}")

        return self.loss_mean

    # one training step / batch
    def train_step(self, input, target):
        # zero grads before training steps
        self.optimizer.zero_grad()
        # compute pde residual
        output = self.forward(input)
        # realign target dimensions
        if self.task == "regression":
            target = target.view(output.shape)
            # compute loss
        loss = self.criterion(output, target)
        # prop loss backwards to
        loss.backward()
        # update parameters
        self.optimizer.step()
        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        # compute correct
        correct = 0
        if self.task == "classification":
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        return loss.item(), correct

    # one training epoch
    @enable_grad()
    def train_epoch(self, trainloader, epoch, idx_out=10):
        logging.info(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        #
        if self.verbosity > 4:
            start = timeit.default_timer()

        # enter loop over batches
        for idx, data in enumerate(trainloader):
            input, target = data
            # send to cuda
            input, target = input.to(self.device), target.to(self.device)

            # take one training step
            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            loss, correct = self.train_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
            # logging
            if idx > 0 and idx % idx_out == 0:
                loss_running = loss_acc / n_data
                if self.task == "classification":
                    accuracy = correct_acc / n_data
                elif self.task == "regression":
                    # use r2
                    accuracy = 1 - loss_running / self.loss_mean

                logging.info(
                    f"epoch {epoch} -batch {idx}/{len(trainloader)} --- running ::: loss: {loss_running}; accuracy: {accuracy} "
                )

        if self.verbosity > 4:
            end = timeit.default_timer()
            print(f"training time for epoch {epoch}: {end-start} seconds")

        self.model.eval()
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean
        return loss_running, accuracy

    # test batch
    def test_step(self, input, target):
        with torch.no_grad():
            # forward pass: prediction
            output = self.forward(input)
            # realign target dimensions
            if self.task == "regression":
                target = target.view(output.shape)
            # compute loss
            loss = self.criterion(output, target)
            correct = 0
            if self.task == "classification":
                # compute correct
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
            return loss.item(), correct

    # test epoch
    def test_epoch(self, testloader, epoch):
        logging.info(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        for idx, data in enumerate(testloader):
            input, target = data
            # send to cuda
            input, target = input.to(self.device), target.to(self.device)
            # perform test step on batch.
            loss, correct = self.test_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean
        logging.info(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy

    # test batch
    def _step(self, input, target):
        # forward pass: prediction
        output = self.forward(input)
        # realign target dimensions
        if self.task == "regression":
            target = target.view(output.shape)
        # compute loss
        loss = self.criterion(output, target)
        correct = 0
        if self.task == "classification":
            # compute correct
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        return loss, correct

    # test epoch
    def _eval(self, testloader):
        logging.info(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        for idx, data in enumerate(testloader):
            input, target = data
            # send to cuda
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            # take one training step
            loss, correct = self._step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean
        logging.info(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy

    def compute_confusion_matrix(self, testloader, nb_classes):
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(testloader):
                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                output = self.model(inputs)
                _, preds = torch.max(output.data, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        return confusion_matrix

    # training loop over all epochs
    def train_loop(self, config, tune=False):
        logging.info("##### enter training loop ####")

        # unpack training_config
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        val_epochs = config["training::val_epochs"]
        idx_out = config["training::idx_out"]
        checkpoint_dir = config["training::checkpoint_dir"]

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]
        # dataloader

        if self.task == "regression":
            self.compute_mean_loss(testloader)

        perf_dict = {
            "train_loss": 1e15,
            "train_accuracy": 0.0,
            "test_loss": 1e15,
            "test_accuracy": 0.0,
        }
        # self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        # enter training loop
        for epoch in epoch_iter:
            # enter training loop over all batches
            loss, accuracy = self.train_epoch(trainloader, epoch, idx_out=idx_out)

            if epoch % val_epochs == 0:
                loss_test, accuracy_test = self.test_epoch(testloader, epoch)

                if loss_test < self.loss_best:
                    self.best_epoch = epoch
                    self.loss_best = loss_test
                    perf_dict["epoch"] = epoch
                    perf_dict["train_loss"] = loss
                    perf_dict["train_accuracy"] = accuracy
                    perf_dict["test_loss"] = loss_test
                    perf_dict["test_accuracy"] = accuracy_test
                    self.save_model(
                        epoch="best", perf_dict=perf_dict, path=checkpoint_dir
                    )
                logging.info(f"best loss: {self.loss_best} at epoch {self.best_epoch}")

            if epoch % output_epoch == 0:
                perf_dict["train_loss"] = loss
                perf_dict["train_accuracy"] = accuracy
                perf_dict["test_loss"] = loss_test
                perf_dict["test_accuracy"] = accuracy_test

        logging.info(f"best loss: {self.loss_best} at epoch {self.best_epoch}")
        return self.loss_best


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
