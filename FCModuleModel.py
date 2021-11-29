#!/usr/bin/python3

import numpy as np
from numpy import genfromtxt
import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import wandb

from dataFCModule import FCModuleDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Logging
use_wandb = True

# Data
num_input_feature = 2
num_joint = 7
sequence_length = 10
num_output = 1
output_joint_idx = 6 # Starts from 0

# Cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training
num_epochs = 400
batch_size = 1000
learning_rate_start = 1e-4
learning_rate_end = 1e-5
betas = [0.9, 0.999]

output_max = genfromtxt('MinMax.csv', delimiter=",")[0]
output_max_weight = num_joint * output_max / np.sum(output_max)
print("Scaling: ",output_max_weight)


train_data = FCModuleDataset('./TrainingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, output_idx=output_joint_idx)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)

validation_data = FCModuleDataset('./ValidationData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, output_idx=output_joint_idx)
validationloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)

test_data = FCModuleDataset('./TestingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, output_idx=output_joint_idx)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)

class PandaFCNet(nn.Module):
    def __init__(self, device):
        super(PandaFCNet, self).__init__()
        
        self._device = device
        self.num_epochs = num_epochs
        self.cur_epoch = 0

        hidden_neurons = 50

        layers_backward = []
        layers_backward.append(nn.Linear(sequence_length*num_input_feature*num_joint, hidden_neurons))
        # layers_backward.append(nn.BatchNorm1d(hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, hidden_neurons))
        # layers_backward.append(nn.BatchNorm1d(hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, num_output))
        
        self.backward_network = nn.Sequential(*layers_backward)

        self._optim = optim.Adam(
            self.parameters(),
            lr=learning_rate_start,
            betas=betas
        )

        # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self._optim,
        #                                 lr_lambda=lambda epoch: 0.95 ** epoch)

        if use_wandb is True:
            wandb.init(project="Panda Residual Real Robot Module", tensorboard=False)

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
            
                backward_loss = nn.L1Loss(reduction='sum')(output_scaling[output_joint_idx]*backward_dyn_predictions, output_scaling[output_joint_idx]*inputs) / inputs.shape[0]

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
            
                backward_loss = nn.L1Loss(reduction='sum')(output_scaling[output_joint_idx]*backward_dyn_predictions, output_scaling[output_joint_idx]*inputs) / inputs.shape[0]
                validation_loss = backward_loss

                validation_losses.append(self._to_numpy(validation_loss))
            print("Validation Loss: ", np.mean(validation_losses))

            if use_wandb is True:
                wandb_dict = dict()
                wandb_dict['Training Loss'] = np.mean(train_losses)
                wandb_dict['Validation Loss'] =  np.mean(validation_losses)
                wandb.log(wandb_dict)

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
PandaFC.fit(trainloader=trainloader, validationloader=validationloader)
PandaFC.eval()

PandaFC.save_checkpoint()
for name, param in PandaFC.state_dict().items():
    name= name.replace(".","_")
    file_name = "./result/" + name + ".txt"
    np.savetxt(file_name, param.data.cpu())

batch_idx = 0
forward_real_arr = np.empty((0, num_input_feature*num_joint), float)
forward_pred_arr = np.empty((0, num_input_feature*num_joint), float)

backward_real_arr = np.empty((0, num_output), float)
backward_pred_arr = np.empty((0, num_output), float)

for conditions, states, inputs in testloader:
    conditions = conditions.to(PandaFC._device)
    states = states.to(PandaFC._device)
    inputs = inputs.to(PandaFC._device)

    backward_dyn_predictions = PandaFC.forward(conditions, states, inputs)

    # Create Figure
    # for i in range(num_joint):
    #     plt.subplot(2, 4, i+1)
    #     plt.plot(100*outputs[:,i], color='r', label='real')
    #     plt.plot(100*predictions[:,i].cpu().detach().numpy(), color='b', label='prediction')
    #     plt.legend()
    # plt.savefig('./result/Figure_' + str(batch_idx)+'.png')
    # plt.clf()

    # forward_real_arr = np.append(forward_real_arr, states.cpu().numpy(), axis=0)
    # forward_pred_arr = np.append(forward_pred_arr, forward_dyna_predictions.cpu().detach().numpy(), axis=0)

    backward_real_arr = np.append(backward_real_arr, inputs.cpu().numpy(), axis=0)
    backward_pred_arr = np.append(backward_pred_arr, backward_dyn_predictions.cpu().detach().numpy(), axis=0)

    # if batch_idx == 0:
    #     traced_script_module = torch.jit.trace(PandaFC.to('cpu'), [conditions.cpu(), states.cpu(), inputs.cpu()])
    #     traced_script_module.save("./model/traced_model.pt")
    #     PandaFC.to(device)

    batch_idx = batch_idx+1

real_file_name = './result/backward_real_' + str(output_joint_idx) + '.csv'
pred_file_name = './result/backward_prediction_' + str(output_joint_idx) + '.csv'
np.savetxt(real_file_name,backward_real_arr)
np.savetxt(pred_file_name,backward_pred_arr)