import numpy as np
from numpy.core.numeric import indices
import torch


class UniformReplayMemory():
    def __init__(self, max_size, input_size, device):
        self.mem_size = max_size
        self.mem_cnt = 0
        self.device = device
        self.state_memory = torch.zeros(
            (self.mem_size, input_size), dtype=torch.float32,
            device=self.device)
        self.next_state_memory = torch.zeros(
            (self.mem_size, input_size), dtype=torch.float32,
            device=self.device)
        self.action_memory = torch.zeros(self.mem_size, dtype=torch.int64,
                                         device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32,
                                         device=self.device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool,
                                           device=self.device)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cnt % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cnt += 1

    def sample(self, batch_size):
        # print(f"Mem cnt: {self.mem_cnt}, Mem size: {self.mem_size}")
        max_mem = min(self.mem_cnt, self.mem_size)

        # batch = np.random.choice(max_mem, batch_size, replace=False)
        # Should be much faster
        batch = np.random.default_rng().choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal


class SegmentTree():
    def __init__(self, size):
        self.max = 0.1  # ? this value is used as default propability in Replay Memory
        self.size = 1 << (size - 1).bit_length()
        self.sum = np.zeros(self.size * 2)
        print(f"Initializing Segment Tree with size {self.size}")

    def total(self):
        return self.sum[1]

    def update_index(self, node, p):
        tree = self.sum
        node += self.size
        tree[node] = p
        self.max = max(self.max, p)
        node //= 2
        while node >= 1:
            tree[node] = tree[node * 2] + tree[node * 2 + 1]
            node //= 2

    def _update_nodes(self, nodes):
        children = nodes * 2 + np.expand_dims([0, 1], axis=1)
        self.sum[nodes] = np.sum(self.sum[children], axis=0)

    def _propagate(self, nodes):
        while nodes[0] != 1:
            nodes = nodes // 2
            nodes = nodes.unique()

            self._update_nodes(nodes)

    def update(self, nodes, p):
        nodes += self.size
        self.max = max(p.max(), self.max)
        self.sum[nodes] = p
        self._propagate(nodes)

    def get(self, p):
        node = 1
        pref = 0
        while node < self.size:
            if self.sum[node * 2] + pref >= p:
                node = node * 2
            else:
                pref += self.sum[node * 2]
                node = node * 2 + 1

        return node - self.size, self.sum[node]

    def _find(self, values):
        nodes = np.ones_like(values, dtype=np.int32)
        while nodes[0] < self.size:
            children = nodes * 2 + np.expand_dims([0, 1], axis=1)
            left_children_values = self.sum[children[0]]
            in_left = np.greater(values, left_children_values).astype(np.int32)
            nodes = children[in_left, np.arange(nodes.size)]
            values -= in_left * left_children_values
        return nodes

    def find(self, values):
        nodes = self._find(values)
        data_nodes = nodes - self.size
        return data_nodes, self.sum[nodes]


class PrioritizedReplayMemory():
    def __init__(self, mem_size, input_size, device):
        self.alpha = 0.6
        self.beta = 0.4
        self.mem_cnt = 0
        self.mem_size = mem_size
        self.tree = SegmentTree(mem_size)
        self.device = device
        self.state_memory = torch.zeros(
            (self.mem_size, input_size), dtype=torch.float32,
            device=self.device)
        self.next_state_memory = torch.zeros(
            (self.mem_size, input_size), dtype=torch.float32,
            device=self.device)
        self.action_memory = torch.zeros(self.mem_size, dtype=torch.int64,
                                         device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32,
                                         device=self.device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool,
                                           device=self.device)
        self.td_error_sum = 0
        self.td_error = torch.zeros(
            self.mem_size, dtype=torch.float32, device=self.device)

    def capacity(self):
        return min(self.mem_cnt, self.mem_size)

    def get_avg_td_error(self):
        if self.capacity() == 0:
            return 0.0

        return self.td_error_sum.item() / self.capacity()

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cnt % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cnt += 1

        self.td_error_sum -= self.td_error[index]
        self.td_error[index] = self.tree.max
        self.td_error_sum += self.td_error[index]

        self.tree.update_index(index, self.tree.max)

    def sample(self, batch_size):
        probs, idxs, states, actions, rewards, next_states, terminal = self._get_samples_from_segments(batch_size)

        probs /= self.tree.total()

        weights = (self.capacity() * probs) ** -self.beta
        weights = (weights / weights.max()).type(torch.float32).to(self.device)

        return idxs, states, actions, rewards, next_states, terminal, weights

    def update_priorities(self, idxs, priorities):
        self.td_error_sum -= self.td_error[idxs].sum()
        self.td_error[idxs] = priorities.to(self.device)
        self.td_error_sum += self.td_error[idxs].sum()

        priorities = np.power(priorities + 1e-100, self.alpha)

        # for idx, p in zip(idxs, priorities):
        #     self.tree.update_index(idx, p)

        # * this is much faster
        self.tree.update(idxs, priorities)

    def _get_samples_from_segments(self, batch_size):
        segment_length = self.tree.total() / batch_size

        idxs = torch.zeros(batch_size, dtype=torch.int64)
        probs = torch.zeros(batch_size, dtype=torch.float32)

        probabilities = np.arange(0, batch_size) * segment_length + \
            np.random.uniform(0.0, segment_length, (batch_size, ))

        for i in range(batch_size):
            idx, prob = self.tree.get(probabilities[i])
            idxs[i] = idx
            probs[i] = prob

        states = self.state_memory[idxs]
        actions = self.action_memory[idxs]
        rewards = self.reward_memory[idxs]
        next_states = self.next_state_memory[idxs]
        terminal = self.terminal_memory[idxs]

        return probs, idxs, states, actions, rewards, next_states, terminal


if __name__ == '__main__':
    memory = PrioritizedReplayMemory(24, 1, 'cpu')
    memory.store_transition(0, 0, 0, 0, False)
    memory.store_transition(1, 0, 0, 0, False)
    memory.store_transition(2, 0, 0, 0, False)
    memory.store_transition(3, 0, 0, 0, False)
    print(memory.sample(3)[0])
    memory.update_priorities(torch.arange(0, 4, dtype=torch.int64), torch.tensor(np.random.uniform(0, 1.0, (4)), dtype=torch.float))
    print(memory.sample(2)[0])
