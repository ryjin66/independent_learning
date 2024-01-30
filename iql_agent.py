import os
import collections

import numpy as np
import torch
import torch.nn as nn


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states)


class Q_net(nn.Module):
    def __init__(self, num_s, num_a, num_hidden):
        super(Q_net, self).__init__()
        self.state_dim = num_s
        self.action_dim = num_a
        self.fc1 = nn.Linear(num_s, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, num_a)

    def forward(self, s):
        s = s.reshape(-1, self.state_dim)
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        q = self.fc4(x)
        return q


class agent():
    def __init__(self,
                 d_s,
                 d_a,
                 gamma,
                 learning_rate,
                 sync_target,
                 max_epsilon,
                 min_epsilon,
                 max_steps,
                 buffer_size,
                 batch_size,
                 replay_start_size,
                 update_every,
                 device):
        super(agent, self).__init__()
        self.num_action = d_a
        self.gamma = gamma
        self.sync_target = sync_target
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = max_epsilon
        self.max_steps = max_steps
        self.alpha = learning_rate
        self.update_every = update_every
        self.batch_size = batch_size
        self.replay_buffer = ExperienceReplay(buffer_size)
        self.replay_start_size = replay_start_size

        # Network and optimizer
        self.Q = Q_net(d_s, d_a, 64).to(device)
        self.target_Q = Q_net(d_s, d_a, 64).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()

        self.device = device
        self.counter = 0

    def select_action(self, state):
        state_a = np.array([state], copy=False)
        state_v = torch.FloatTensor(state_a).to(self.device)
        action_value = self.Q(state_v)
        _, act_v = torch.max(action_value, dim=1)
        action = int(act_v.item())
        return action

    def step(self, observation, is_evaluation=False):
        if is_evaluation:
            with torch.no_grad():
                action = self.select_action(observation)
        else:
            self.counter += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (
                max(self.max_steps - self.counter, 0)) / self.max_steps
            if np.random.random() <= self.epsilon:
                action = int(np.random.choice(self.num_action))
            else:
                action = self.select_action(observation)
            if self.counter % self.update_every == 0 and len(self.replay_buffer) >= self.replay_start_size:
                self.learn()
            if self.counter % self.sync_target == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())
        return action

    def learn(self):
        # sampling from replay buffer
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        states_v = torch.FloatTensor(states).to(self.device)
        next_states_v = torch.FloatTensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        # Q(s_t,a_t) from self.Q() net
        state_action_values = self.Q(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        # r+gamma * max_{a'} Q(s_{t+1},a')
        next_state_value = self.target_Q(next_states_v).max(1)[0]
        next_state_value = next_state_value.detach()
        expected_state_action_values = rewards_v + self.gamma * next_state_value
        # form Q-loss and perform back propagation
        Q_loss = self.loss(state_action_values, expected_state_action_values)
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

    def save(self, case_num, episode=0, directory='./model/'):
        path = directory + 'policy{}_{}.pth'.format(case_num, episode)
        if os.path.exists(directory):
            torch.save(self.Q.state_dict(), path)
            print("====================================")
            print("Model has been saved...")
            print("====================================")
        else:
            print("Error, no such directory!")

    def load(self, directory='./model/'):
        # path = directory + self.player_id + 'policy{}.pth'.format(case_num)
        self.Q.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
        self.target_Q.load_state_dict(torch.load(directory, map_location=torch.device('cpu')))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
