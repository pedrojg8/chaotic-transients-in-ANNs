#!/usr/bin/env python3

"""
MNIST_tanh.py 

WHAT THIS SCRIPT IS FOR:
  - Generate a reference training trajectory (in parameter space) and multiple
    perturbed trajectories that start from (almost) the same initial condition.
  - Measure how the distance d(t) between trajectories grows during training.

HIGH-LEVEL PIPELINE:
  1) Load MNIST (train/test). Use full-batch training to avoid mini-batch noise.
  2) Define a baseline MLP with tanh.
  3) For each random initialization i = 1..n_init:
       a) Train the reference network and record the flattened parameter vector w(t)
          at each epoch (trajectory in weight space).
       b) Create num_pert perturbed networks by adding tiny noise ±eps to weights
          (bias excluded), train each, and record w_pert(t).
       c) Compute L1 distance d(t) = ||w(t) - w_pert(t)||_1 for all perturbations.
  4) Save all distance trajectories to an HDF5 file (results/...) for later fitting.

IMPORTANT NOTES:
  - This script expects to be launched as a SLURM array job:
        SLURM_ARRAY_TASK_ID selects the learning rate from learning_rate_array.
    If you run locally without SLURM, this env var will be missing and the script
    will crash unless you set it manually.
  - The output file is written under the folder "results/". 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import powerlaw
import networkx as nx
from scipy import stats
from torch.nn.utils import parameters_to_vector
import os
from torchvision import datasets, transforms
import time

from matplotlib import rcParams

rcParams["font.size"] = 18
rcParams["axes.labelsize"] = 24

rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14

rcParams["figure.figsize"] = (8,6)

# Define transformations for the dataset (without normalization)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Convert datasets to DataLoader for batch processing
# NOTE: batch_size=len(dataset) => 1 batch per epoch => full-batch training.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# ----------------------------
# Model definition (MLP tanh)
# ----------------------------
# Purpose:
#   - Simple 2-layer MLP for MNIST.
#   - tanh nonlinearity (paper baseline).
#   - double precision layers (.double()) to allow extremely small eps (e.g. 1e-8).

class MNISTNet(nn.Module):
    def __init__(self, input_dim=784, hidden_units=64, output_dim=10, initial_state=None):
        super(MNISTNet, self).__init__()
        # Create fully connected layers in double precision.
        self.fc1 = nn.Linear(input_dim, hidden_units).double()
        self.fc2 = nn.Linear(hidden_units, output_dim).double()
        
        # If an initial_state is provided, we load it (for perturbed replicas).
        # Otherwise, initialize weights from the baseline distribution.
        if initial_state is not None:
            self.load_state_dict(initial_state)
        else:
            self.init_weights()
    
    def init_weights(self):
        # Paper baseline:
        #   - weights ~ Normal(0,1)
        #   - biases = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=1)
                module.bias.data.fill_(0)
    
    def forward(self, x):
        # Flatten the input (MNIST images are 28x28 → 784 features)
        x = x.view(-1, 784)
        # Hidden layer + tanh
        x = torch.tanh(self.fc1(x))
        # Output logits
        x = self.fc2(x)
        return x
    
######################################
# Helper Functions: Flattening & L1 Distance
######################################

# Purpose:
#   - For Lyapunov-style analysis we treat the parameter vector w(t) as the "state"
#     of the dynamical system. We therefore flatten all trainable parameters
#     into a single vector at each epoch.

def get_flat_params(model):
    # parameters_to_vector concatenates all parameters into a 1D tensor.
    # clone() to avoid accidental aliasing
    return parameters_to_vector(model.parameters()).clone()

######################################
# Training Functions that Record Flattened Parameter Trajectories
######################################

# Purpose:
#   - Train a reference network from scratch.
#   - Record trajectory w(t) (flattened params) at each epoch, including t=0.
#   - Return the initial_state (state_dict) so perturbed replicas start identically.

def initialize_network_and_flatten(num_epochs, learning_rate, input_dim, hidden_units, output_dim, initial_weights=None):
    net = MNISTNet(input_dim, hidden_units, output_dim, initial_weights)
    
    for param in net.parameters():
        param.data = param.data.double()
    
    # Save exact initial condition to reuse for perturbed runs.
    initial_state = {k: v.clone() for k, v in net.state_dict().items()}
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)

    losses = np.zeros(num_epochs)

    # flat_params[t] will store w(t). We include t=0 before any training step.
    flat_params = []
    flat_params.append(get_flat_params(net))
    
    # Full-batch GD: there is exactly one batch in train_loader.
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.double())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        losses[epoch] = loss.item()
        flat_params.append(get_flat_params(net))
    
    return initial_state, losses, flat_params

# Purpose:
#   - Create num_pert replicas starting from the SAME initial_state.
#   - Apply a small perturbation to weights (excluding biases).
#   - Train each perturbed network, recording w_pert(t) for divergence analysis.

def pert_initial_conditions_flat(num_epochs, learning_rate, initial_state, eps, num_pert, input_dim, hidden_units, output_dim):
    pert_losses = []
    pert_flat_params = []
    
    for j in range(num_pert):
        # Start from the exact same initial_state as reference
        net_pert = MNISTNet(input_dim, hidden_units, output_dim, initial_state)

        for param in net_pert.parameters():
            param.data = param.data.double()

        # Apply a small random perturbation to each weight parameter:
        #   param <- param + U(-eps, eps)
        # Biases are excluded
        with torch.no_grad():
            for name, param in net_pert.named_parameters():
                if name.endswith("bias"):
                    continue
                else:
                    param.add_(torch.rand_like(param) * 2 * eps - eps)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net_pert.parameters(), lr=learning_rate, momentum=0)
        losses = np.zeros(num_epochs)
        flat_params = []
        flat_params.append(get_flat_params(net_pert))  # t=0
        
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = net_pert(inputs.double())
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            losses[epoch] = loss.item()
            flat_params.append(get_flat_params(net_pert))
        
        pert_losses.append(losses)
        pert_flat_params.append(flat_params)
    
    return pert_losses, pert_flat_params

# Purpose:
#   - Compute the divergence trajectory d(t) between reference and perturbed runs.
#   - Here: L1 norm in parameter space: d(t) = sum_k |w_k(t) - w'_k(t)|.
#   - Output is a Python list of length num_epochs+1.

def distance_trajectories_flat(flat_params_orig, flat_params_pert):
    distances = []
    for vec_orig, vec_pert in zip(flat_params_orig, flat_params_pert):
        distances.append((vec_orig - vec_pert).abs().sum().item())
    return distances 

######################################
# Main routine: store distance trajectories (Lyapunov-style data)
######################################

# Purpose:
#   - For a given learning_rate:
#       run n_init independent initializations
#       for each, train a reference + num_pert perturbed replicas
#       store all distance trajectories into HDF5
#
# Output structure:
#   weights[eps_idx, init_idx, pert_idx, t] = d(t)

def lyapunov_exponent(learning_rate):
    num_epochs = 201
    n_init = 50

    # You can use multiple eps values; here a single eps is used.
    epsilon = np.array([1e-8])
    num_pert = 5

    input_dim = 784
    hidden_units = 64
    output_dim = 10

    # Dataset shape for storage
    w_shape = (len(epsilon), n_init, num_pert, num_epochs + 1)

    tic = time.time()
    print("Starting Lyapunov exponent computation...")

    with h5py.File(f'results/MNIST_lyapunov_gpt_tanh_lr_{learning_rate}.h5', 'a') as f:
        # Save metadata
        attrs_dict = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'n_init': n_init,
            'num_pert': num_pert,
            'epsilon': epsilon.tolist()
        }
        for k, v in attrs_dict.items():
            f.attrs[k] = v

        # Create the dataset to store distances
        w_ds = f.require_dataset('weights', w_shape, dtype=np.float64)
        
        for eps_idx, eps in enumerate(epsilon):
            for i in range(n_init):
                # (1) Reference run
                initial_state, _, flat_traj_orig = initialize_network_and_flatten(num_epochs, learning_rate,
                                                                input_dim, hidden_units, output_dim)
                
                # (2) Perturbed runs (same initial_state + tiny perturbation)
                _, pert_flat_trajs = pert_initial_conditions_flat(num_epochs, learning_rate,
                                                                initial_state, eps, num_pert,
                                                                input_dim, hidden_units, output_dim)
                
                # (3) Compute distance trajectories d(t) for each perturbation
                distances = [distance_trajectories_flat(flat_traj_orig, pert_traj)
                            for pert_traj in pert_flat_trajs]
                
                # (4) Save: shape (num_pert, num_epochs+1)
                w_ds[eps_idx, i, :, :] = np.array(distances)

            toc = time.time()
            print(f"Finished eps index {eps_idx+1}/{len(epsilon)} after {toc-tic:.2f}s")
            
    toc = time.time()
    print(f"Finished in {toc-tic:.2f}s")


def main():
    # Purpose:
    #   Sweep learning rates via SLURM array jobs.
    #   Each SLURM task picks one learning rate from this array.
    learning_rate_array = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20]

    # SLURM array index selects which lr to run.
    slurm_array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
  
    learning_rate = learning_rate_array[slurm_array_task_id]

    lyapunov_exponent(learning_rate)


if __name__ == "__main__":
    main()