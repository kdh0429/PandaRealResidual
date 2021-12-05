#!/usr/bin/python3

import numpy as np
from numpy import genfromtxt
import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from dataFC import FCDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# Data
num_input_feature = 2
num_joint = 7
sequence_length = 10

# Training
num_epochs = 2500
batch_size = 1000
learning_rate_start = 1e-3
learning_rate_end = 1e-4
betas = [0.9, 0.999]

# Cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


output_max = genfromtxt('./data/MinMax.csv', delimiter=",")[0]
output_max_weight = num_joint * output_max / np.sum(output_max)
print("Scaling: ",output_max_weight)


class PandaFCNet(nn.Module):
    def __init__(self, device):
        super(PandaFCNet, self).__init__()
        
        self._device = device
        self.num_epochs = num_epochs
        self.cur_epoch = 0

        hidden_neurons = 100

        layers_backward = []
        layers_backward.append(nn.Linear(sequence_length*num_input_feature*num_joint, hidden_neurons))
        # layers_backward.append(nn.BatchNorm1d(hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, hidden_neurons))
        # layers_backward.append(nn.BatchNorm1d(hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, num_joint))
        
        self.backward_network = nn.Sequential(*layers_backward)

        self._optim = optim.Adam(
            self.parameters(),
            lr=learning_rate_start,
            betas=betas
        )

        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self._optim,
        #                                 lr_lambda=lambda epoch: 0.95 ** epoch)


    def forward(self, condition, state, input):
        backward_output = self.backward_network(torch.cat([condition, state], dim=1))
        return backward_output

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, trainloader, validationloader, print_every=1):
        """
        Train the neural network
        """

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1
            # if epoch == self.num_epochs/2:
            for param_group in self._optim.param_groups:
                param_group['lr'] = learning_rate_start * math.exp(math.log(learning_rate_end/ learning_rate_start) * epoch / num_epochs)

            train_losses = []
            
            for conditions, states, inputs in trainloader:
                self.train()
                self._optim.zero_grad()

                conditions = conditions.to(self._device)
                states = states.to(self._device)
                inputs = inputs.to(self._device)

                output_scaling = torch.from_numpy(output_max_weight).to(self._device)

                backward_dyn_predictions = self.forward(conditions, states, inputs)
            
                backward_loss = nn.L1Loss(reduction='sum')(output_scaling*backward_dyn_predictions, output_scaling*inputs) / inputs.shape[0]

                train_loss = backward_loss
                train_loss.backward()

                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))

            # self.scheduler.step()

            print('Training Loss: ', np.mean(train_losses))

            validation_losses = []
            self.eval()
            for conditions, states, inputs in validationloader:
                conditions = conditions.to(self._device)
                states = states.to(self._device)
                inputs = inputs.to(self._device)

                backward_dyn_predictions = self.forward(conditions, states, inputs)
            
                backward_loss = nn.L1Loss(reduction='sum')(output_scaling*backward_dyn_predictions, output_scaling*inputs) / inputs.shape[0]
                validation_loss = backward_loss

                validation_losses.append(self._to_numpy(validation_loss))
            print("Validation Loss: ", np.mean(validation_losses))


    def save_checkpoint(self):
        """Save model paramers under config['model_path']"""
        model_path = './model/pytorch_model.pt'

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, model_path):
        """
        Retore the model parameters
        """
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])


PandaFC = PandaFCNet(device)
PandaFC.to(device)
PandaFC.restore_model('./model/pytorch_model.pt')
PandaFC.eval()

Deg2Rad = 0.0174532
q=[0.00600409,  -0.0041471, -0.00431325,    -1.65806,  -0.0242243,     1.63892,   -0.050714]
qdot = [-8.29321e-05, -0.000171542,  -8.3235e-05, -0.000452614,  7.70033e-05,  1.99693e-05,  7.51343e-05]
conditions = 9*[q[0], qdot[0], q[1], qdot[1], q[2], qdot[2], q[3], qdot[3], q[4], qdot[4], q[5], qdot[5],  q[6], qdot[6]]
states = [q[0], qdot[0], q[1], qdot[1], q[2], qdot[2], q[3], qdot[3], q[4], qdot[4], q[5], qdot[5],  q[6], qdot[6]]
inputs = [-0.323497,   -26.7984,  -0.111344,    20.3115,   0.954648,    2.01611, -0.0181386] #[-0.101329, -26.7984, -0.210526, 20.3115, 0.831052, 1.78814, 0.0404552]

inputs = np.array(inputs) / output_max
conditions = np.array(conditions) / np.tile(np.array([3.14, 0.5]), (1,7*9))
states = np.array(states) / np.tile(np.array([3.14, 0.5]), (1,7))


inputs=torch.from_numpy(inputs)
conditions=torch.from_numpy(conditions)
states=torch.from_numpy(states)

conditions = conditions.to(PandaFC._device, dtype=torch.float)
states = states.to(PandaFC._device, dtype=torch.float)
inputs = inputs.to(PandaFC._device, dtype=torch.float)

backward_dyn_predictions = PandaFC.forward(conditions, states, inputs)
print("Network output: ", backward_dyn_predictions.cpu().detach().numpy())
print(output_max*(inputs.cpu().numpy() - backward_dyn_predictions.cpu().detach().numpy()))
