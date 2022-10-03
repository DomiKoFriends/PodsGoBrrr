import numpy as np
import torch

import base65536
from memory import UniformReplayMemory, PrioritizedReplayMemory
from model import DQN, DuelingDQN

from os import makedirs, path


class AbstractAgent():
    def __init__(self, name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions, input_dim, mem_size, batch_size, device, clip_value):
        params = locals()

        directory_path = path.join(".", "models", name, "model_params")
        if not path.exists(directory_path):
            print(f"Creating directory: {directory_path}")
            makedirs(directory_path)

        file_path = path.join(directory_path, "initial_parameters.txt")
        with open(file_path, 'w') as file:
            print(params, file=file)

        directory_path = path.join(".", "models", name, "checkpoints")

        if not path.exists(directory_path):
            print(f"Creating directory: {directory_path}")
            makedirs(directory_path)

        directory_path = path.join(".", "models", name, "weights")

        if not path.exists(directory_path):
            print(f"Creating directory: {directory_path}")
            makedirs(directory_path)

        self.name = name
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_steps = 0
        self.lr = lr
        self.n_actions = n_actions
        self.input_dim = input_dim
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.device = device
        self.memory = None
        self.action_space = [i for i in range(self.n_actions)]

        self.net = None
        self.frozen_net = None

    def set_net(self, net):
        self.net = net

        file_path = path.join(".", "models", self.name,
                              "model_params", "initial_parameters.txt")
        if not path.exists(file_path):
            raise Exception(
                "Initial parameters description file doesn\'t exist!")

        with open(file_path, 'a') as file:
            print(self.net.layers, file=file)

    def act(self, X):
        raise NotImplementedError("Abstract agent act method not implemented")

    def learn(self):
        raise NotImplementedError(
            "Abstract agent learn method not implemented")

    def act_e_greedy(self, X):
        if np.random.random() > self.epsilon:
            return self.act(X)
        else:
            action = np.random.choice(self.action_space)

        return action

    def evaluate_q(self, state):
        raise NotImplementedError(
            "Abstract agent evaluate_q method not implemented")

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self, steps):
        self.net.save_checkpoint(steps)
        self.save_weights(steps)

    def load_models(self, steps, model_name=None, load_optimizer=False):
        self.net.load_checkpoint(steps, model_name, load_optimizer)
        self.update_frozen_net()

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def update_frozen_net(self):
        self.frozen_net.load_state_dict(self.net.state_dict())

    def decrease_epsilon(self):
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        self.epsilon_steps += 1
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
            np.exp(-1 * self.epsilon_steps / self.epsilon_decay)

    def save_weights(self, steps):

        def compress(weights):
            array = list(weights.reshape(-1))
            array_string = bytes(" ".join(str(i)
                                 for i in array), encoding="ascii")
            small = base65536.encode(array_string)
            return small

        file_name = path.join(".", "models", self.name, "weights", f"{steps}")
        with open(file_name, 'w') as file:
            weights, biases = [], []
            for name, param in self.net.named_parameters():
                if 'weight' in name:
                    weights.append(compress(param.detach().cpu().numpy()))
                else:
                    biases.append(compress(param.detach().cpu().numpy()))

            print('w = ', weights, file=file)
            print('b = ', biases, file=file)


class DQNAgent(AbstractAgent):
    def __init__(self, agent_name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions, input_dim, mem_size, batch_size, device, clip_value=None):
        super(DQNAgent, self).__init__(agent_name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions,
                                       input_dim, mem_size, batch_size, device, clip_value)
        self.set_net(DQN(self.lr, self.n_actions, input_dim,
                         path.join(".", "models", agent_name, "checkpoints"), device))

        self.memory = UniformReplayMemory(mem_size, input_dim, device)

    def act(self, X):
        with torch.no_grad():
            q_val = self.net.forward(X)
            action = torch.argmax(q_val).item()
            return action

    def act_many(self, X):
        with torch.no_grad():
            q_val = self.net.forward(X)
            actions = torch.argmax(q_val, dim=1)
            return actions

    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return

        self.net.optimizer.zero_grad()

        state, action, reward, new_state, terminal = self.memory.sample(
            self.batch_size)

        indices = torch.arange(0, self.batch_size)

        q_val = self.net.forward(state)[indices, action]
        q_next = self.net.forward(new_state)
        q_next[terminal] = 0.0

        q_target = reward + self.gamma * q_next.max(dim=1)[0]

        loss = self.net.loss(q_target, q_val).to(self.device)
        loss.backward()

        # Gradient Value Clipping
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.net.parameters(), clip_value=self.clip_value)

        self.net.optimizer.step()

    def evaluate_q(self, state):
        with torch.no_grad():
            q_val = self.net.forward(state)
            return q_val.max(dim=1)[0]


class DoubleDQNAgent(AbstractAgent):
    def __init__(self, agent_name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions, input_dim, mem_size, batch_size, device, clip_value=None):
        super(DoubleDQNAgent, self).__init__(agent_name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions,
                                             input_dim, mem_size, batch_size, device, clip_value)

        self.set_net(DQN(lr, n_actions, input_dim, path.join(
            ".", "models", agent_name, "checkpoints"), device))
        self.frozen_net = DQN(lr, n_actions, input_dim, path.join(
            ".", "models", agent_name, "checkpoints"), device)

        self.update_frozen_net()
        self.memory = UniformReplayMemory(mem_size, input_dim, device)

    def act(self, X):
        with torch.no_grad():
            q_val = self.net.forward(X)
            action = torch.argmax(q_val).item()
            return action

    def act_many(self, X):
        with torch.no_grad():
            q_val = self.net.forward(X)
            actions = torch.argmax(q_val, dim=1)
            return actions

    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return

        self.net.optimizer.zero_grad()

        state, action, reward, new_state, terminal = self.memory.sample(
            self.batch_size)

        indices = torch.arange(0, self.batch_size)

        q_val = self.net.forward(state)[indices, action]

        q_next = self.frozen_net.forward(new_state)
        q_next[terminal] = 0.0

        q_eval = self.net.forward(new_state)
        target_actions = torch.argmax(q_eval, dim=1)

        q_target = reward + self.gamma * q_next[indices, target_actions]

        loss = self.net.loss(q_target, q_val).to(self.device)
        loss.backward()

        # Gradient Value Clipping
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.net.parameters(), clip_value=self.clip_value)

        self.net.optimizer.step()

    def evaluate_q(self, state):
        with torch.no_grad():
            q_val = self.net.forward(state)
            return q_val.max(dim=1)[0]


class DoubleDQNAgentPrioritized(AbstractAgent):
    def __init__(self, agent_name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions, input_dim, mem_size, batch_size, device, clip_value=None):
        super(DoubleDQNAgentPrioritized, self).__init__(agent_name, gamma, epsilon_start, epsilon_min, epsilon_decay, lr, n_actions,
                                                        input_dim, mem_size, batch_size, device, clip_value)

        self.set_net(DQN(lr, n_actions, input_dim, path.join(
            ".", "models", agent_name, "checkpoints"), device))

        self.frozen_net = DQN(lr, n_actions, input_dim, path.join(
            ".", "models", agent_name, "checkpoints"), device)

        self.update_frozen_net()
        self.memory = PrioritizedReplayMemory(mem_size, input_dim, device)

    def act(self, X):
        with torch.no_grad():
            q_val = self.net.forward(X)
            action = torch.argmax(q_val).item()
            return action

    def act_many(self, X):
        with torch.no_grad():
            q_val = self.net.forward(X)
            actions = torch.argmax(q_val, dim=1)
            return actions

    def act_many_e_greedy(self, X):
        if np.random.random() > self.epsilon:
            return self.act_many(X)
        else:
            return np.random.choice(self.action_space, size=(X.shape[0]))

    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return

        self.net.optimizer.zero_grad()

        idxs, state, action, reward, new_state, terminal, weights = self.memory.sample(
            self.batch_size)

        indices = torch.arange(0, self.batch_size)

        q_val = self.net.forward(state)[indices, action]

        q_next = self.frozen_net.forward(new_state)
        q_next[terminal] = 0.0

        q_eval = self.net.forward(new_state)
        target_actions = torch.argmax(q_eval, dim=1)

        q_target = reward + self.gamma * q_next[indices, target_actions]

        loss = (weights * (q_val - q_target) ** 2).mean()
        self.memory.update_priorities(
            idxs, (q_val - q_target).abs().detach().cpu())
        loss.to(self.device)

        # loss *= weights
        loss.backward()

        # Gradient Value Clipping
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.net.parameters(), clip_value=self.clip_value)

        self.net.optimizer.step()

    def evaluate_q(self, state):
        with torch.no_grad():
            q_val = self.net.forward(state)
            return q_val.max(dim=1)[0]


class DuelingDoubleDQNAgent(AbstractAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dim, mem_size, batch_size, epsilon_decay, name, device, epsilon_min=0.01):
        super(DuelingDoubleDQNAgent, self).__init__(gamma, epsilon, lr, n_actions,
                                                    input_dim, mem_size, batch_size, device, epsilon_min, epsilon_decay)

        self.net = DuelingDQN(
            lr, n_actions, name + '_DuelingDoubleDQN_VAL', input_dim, 'models', device)

        self.frozen_net = DuelingDQN(
            lr, n_actions, name + '_DuelingDoubleDQN_frozen', input_dim, 'models', device)

        self.update_frozen_net()
        self.memory = UniformReplayMemory(mem_size, input_dim, device)

    def act(self, X):
        with torch.no_grad():
            _, advantage = self.net.forward(X)
            action = torch.argmax(advantage).item()
            return action

    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return

        self.net.optimizer.zero_grad()

        state, action, reward, new_state, terminal = self.memory.sample(
            self.batch_size)

        indices = torch.arange(0, self.batch_size)

        V_state, A_state, = self.net.forward(state)
        V_next_state, A_next_state = self.frozen_net.forward(state)

        V_state_eval, A_state_eval = self.net.forward(new_state)

        q_pred = torch.add(
            V_state, (A_state - A_state.mean(dim=1, keepdim=True)))[indices, action]
        q_next = torch.add(V_next_state, (A_next_state -
                           A_next_state.mean(dim=1, keepdim=True)))
        q_eval = torch.add(V_state_eval, (A_state_eval -
                           A_state_eval.mean(dim=1, keepdim=True)))

        target_action = torch.argmax(q_eval, dim=1)

        q_next[terminal] = 0.0
        q_target = reward + self.gamma * q_next[indices, target_action]
        loss = self.net.loss(q_target, q_pred).to(self.device)
        loss.backward()

        # Gradient Value Clipping
        # torch.nn.utils.clip_grad_value_(
        #     self.net.parameters(), clip_value=5.0)

        # Gradient Norm Clipping
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        self.net.optimizer.step()

    def evaluate_q(self, state):
        with torch.no_grad():
            v, a = self.net.forward(state)
            return (torch.add(v, (a - a.mean(dim=1, keepdim=True)))).max(dim=1)[0]
