from numpy import result_type
import torch
import torch.nn.functional as F
from torch import nn
import os
from os import path

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def forward(self, X):
        return self.layers.forward(X)

    def loss(self, Out, Targets):
        return F.mse_loss(Out, Targets)


class DQN(nn.Module):
    def __init__(self, lr, n_actions, input_dim, checkpoint_dir, device):
        super(DQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions)
        )

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def forward(self, X):
        result = self.layers(X)
        return result

    def save_checkpoint(self, steps):
        print(f"Saving network 😱")
        torch.save(self.state_dict(), path.join(self.checkpoint_dir, f"net_{steps}"))
        torch.save(self.optimizer.state_dict(), path.join(self.checkpoint_dir, f"optimizer_{steps}"))

    def load_checkpoint(self, steps, model_name, load_optimizer):
        print(f"Loading network 👻")
        if model_name is None:
            self.load_state_dict(torch.load(path.join(self.checkpoint_dir, f"net_{steps}")))
            if load_optimizer:
                self.optimizer.load_state_dict(torch.load(path.join(self.checkpoint_dir, f"optimizer_{steps}")))
        else:
            directory = path.join(".", "models", model_name, "checkpoints")
            self.load_state_dict(torch.load(path.join(directory, f"net_{steps}")))
            if load_optimizer:
                self.optimizer.load_state_dict(torch.load(path.join(directory, f"optimizer_{steps}")))

class DuelingDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dim, checkpoint_dir, device):
        super(DuelingDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.name = name
        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        
        self.V = nn.Linear(16, 1)
        self.A = nn.Linear(16, n_actions)
    
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def forward(self, X):
        result = self.layers(X)

        V = self.V(result)
        A = self.A(result)

        return V, A

    def save_checkpoint(self, steps):
        print(f"Saving network: {self.name}")
        torch.save(self.state_dict(), self.checkpoint_file + f"_{steps}")

    def load_checkpoint(self):
        print(f"Loading network: {self.name}")
        self.load_state_dict(torch.load(self.checkpoint_file))
