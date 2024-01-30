import numpy as np
import argparse
import matplotlib.pyplot as plt


class artificial_env():
    def __init__(self):
        self.state = None
        self.state_space = [0, 1]

    def initialize(self):
        s = [[1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1]]
        self.state = s[np.random.choice(4, p=[0.25, 0.25, 0.25, 0.25])]
        return self.state, 0

    def step(self, action):
        old_state = self.state.copy()
        if action[0] != action[1]:
            self.state[0] = 1 - self.state[0]
            self.state[1] = 1 - self.state[1]

        if self.state[2] == 1:
            self.state[2] = np.random.choice(self.state_space, p=[0.5, 0.5])
        else:
            if action[1] != action[2]:
                self.state[2] = np.random.choice(self.state_space, p=[0, 1])
            else:
                self.state[2] = np.random.choice(self.state_space, p=[0.5, 0.5])
        rewards = [1 if old_state[i] == self.state[i] else 0 for i in range(3)]

        return rewards, self.state


class INACAgent:
    def __init__(self, max_k, alpha_0, eta_0, mode=0):
        self.action_space1 = [[0, 0], [1, 1], [1, 0], [0, 1]]
        self.action_space2 = [0, 1]
        self.dic = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.Q1 = np.zeros((4, 4))
        self.Q2 = np.zeros((2, 2))
        self.theta1 = np.zeros((4, 4))
        self.theta2 = np.zeros((2, 2))
        self.last_state = None
        self.last_action = None
        self.max_k = max_k
        self.k = 0
        self.t = 0
        self.gamma = 0.99
        self.alpha = alpha_0
        self.eta = eta_0
        self.eta_sum = eta_0
        self.option = [[1, 2], [0, 2], [0, 1]]
        self.mode = mode

    def get_key(self, value):
        return [k for k, v in self.dic.items() if v == value]

    def get_split(self, sv):
        sv1 = [int(sv[i]) for i in self.option[self.mode]]
        sv1 = int(self.get_key(sv1)[0])
        sv2 = int(sv[self.mode])
        return sv1, sv2

    def get_action(self, state, get_prob=False):
        s1, s2 = self.get_split(state)
        med1 = self.theta1[s1, :]
        med2 = self.theta2[s2, :]
        prob1 = np.exp(med1) / np.sum(np.exp(med1))
        prob2 = np.exp(med2) / np.sum(np.exp(med2))
        if get_prob:
            return prob1, prob2
        action = self.dic[np.random.choice(4, p=prob1)].copy()
        action2 = np.random.choice(2, p=prob2)
        action.insert(self.mode, action2)
        return action

    def step(self, reward, state, is_test=False):
        self.k += 1
        if is_test:
            action = self.get_action(state)
        else:
            random_prob = (1 - self.k / self.max_k) * 0.1
            if np.random.random() < random_prob:
                action = [np.random.choice(2) for i in range(3)]
            else:
                action = self.get_action(state)
            if self.last_action is not None:
                self.q_update(reward, state)
                if self.k == self.max_k:
                    self.policy_update()
                    self.Q1 = np.zeros((4, 4))
                    self.Q2 = np.zeros((2, 2))
                    self.last_state = None
                    self.last_action = None
                    self.k = 0
                    self.t += 1
                    return action
        self.last_state = state
        self.last_action = action
        return action

    def policy_update(self):
        self.eta = self.eta_sum / pow(self.gamma, 2 * self.t - 1)
        self.theta1 = self.theta1 + self.eta * self.Q1
        self.theta2 = self.theta2 + self.eta * self.Q2
        mean1 = np.mean(abs(self.theta1), axis=1) + 0.0001
        mean2 = np.mean(abs(self.theta2), axis=1) + 0.0001
        mean1 = np.tile(mean1, (4, 1)).T
        mean2 = np.tile(mean2, (2, 1)).T
        self.theta1 = self.theta1 / mean1
        # print(self.theta1,mean1)
        self.theta2 = self.theta2 / mean2
        self.eta_sum += self.eta

    def q_update(self, reward, new_state):
        alpha_k = self.alpha / (self.k + 4 * self.alpha)
        s1, s2 = self.get_split(self.last_state)
        a1, a2 = self.get_split(self.last_action)
        new_s1, new_s2 = self.get_split(new_state)
        r1 = sum([reward[i] for i in self.option[self.mode]])
        r2 = reward[self.mode]
        prob1, prob2 = self.get_action(new_state, get_prob=True)
        self.Q1[s1, a1] = (1 - alpha_k) * self.Q1[s1, a1] + alpha_k * (
                r1 + self.gamma * np.dot(self.Q1[new_s1, :], prob1))
        self.Q2[s2, a2] = (1 - alpha_k) * self.Q2[s2, a2] + alpha_k * (
                r2 + self.gamma * np.dot(self.Q2[new_s2, :], prob2))


class IQLAgent:
    def __init__(self, max_step, alpha_0, mode=0):
        self.action_space1 = [[0, 0], [1, 1], [1, 0], [0, 1]]
        self.action_space2 = [0, 1]
        self.dic = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        self.Q1 = np.zeros((4, 4))
        self.Q2 = np.zeros((2, 2))
        self.last_state = None
        self.last_action = None
        self.max_step = max_step
        self.k = 0
        self.gamma = 0.99
        self.alpha = alpha_0
        self.option = [[1, 2], [0, 2], [0, 1]]
        self.mode = mode

    def get_key(self, value):
        return [k for k, v in self.dic.items() if v == value]

    def get_split(self, sv):
        # return the order of state or action
        sv1 = [int(sv[i]) for i in self.option[self.mode]]
        sv1 = int(self.get_key(sv1)[0])
        sv2 = int(sv[self.mode])
        return sv1, sv2

    def get_action(self, state):
        s1, s2 = self.get_split(state)
        action1_index = np.argmax(self.Q1[s1, :])
        action1 = self.dic[action1_index].copy()
        action2 = np.argmax(self.Q2[s2, :])
        action1.insert(self.mode, action2)
        return action1

    def step(self, reward, state, is_test=False):
        if is_test:
            action = self.get_action(state)
        else:
            random_prob = 1 # (1 - self.k / self.max_step)  You can also use decreasing exploration rate
            if np.random.random() < random_prob:
                action = [np.random.choice(2) for i in range(3)]
            else:
                action = self.get_action(state)
            if self.last_action is not None:
                self.q_update(reward, state)
            self.last_state = state
            self.last_action = action
            self.k += 1
        return action

    def q_update(self, reward, state):
        alpha_k = self.alpha / (self.k + 4 * self.alpha)
        s1, s2 = self.get_split(self.last_state)
        a1, a2 = self.get_split(self.last_action)
        new_s1, new_s2 = self.get_split(state)
        new_a1 = np.argmax(self.Q1[new_s1, :])
        new_a2 = np.argmax(self.Q2[new_s2, :])
        r1 = sum([reward[i] for i in self.option[self.mode]])
        r2 = reward[self.mode]
        self.Q1[s1, a1] = (1 - alpha_k) * self.Q1[s1, a1] + alpha_k * (
                r1 + self.gamma * self.Q1[new_s1, new_a1])
        self.Q2[s2, a2] = (1 - alpha_k) * self.Q2[s2, a2] + alpha_k * (
                r2 + self.gamma * self.Q2[new_s2, new_a2])


def get_parser():
    para = argparse.ArgumentParser()
    para.add_argument('--iql_agent', type=int, default=1)
    para.add_argument('--max_step', type=int, default=3000)
    para.add_argument('--test_step', type=int, default=1000)
    para.add_argument('--max_k', type=int, default=100)
    para.add_argument('--alpha_0', type=float, default=0.02)
    para.add_argument('--eta_0', type=float, default=0.2)
    para.add_argument('--num_repeat', type=int, default=1000)
    para.add_argument('--mode', type=int, default=2)
    return para.parse_args()


def run(parser):
    env = artificial_env()
    state, reward = env.initialize()
    episode_reward = 0
    reward_list = []
    if parser.iql_agent == 1:
        agent = IQLAgent(parser.max_step, parser.alpha_0, parser.mode)
        for i in range(1, parser.max_step + 1):
            action = agent.step(reward, state)
            reward, state = env.step(action)
            episode_reward += sum(reward)
            if i % parser.max_k == 0:
                reward_list.append(episode_reward)
                episode_reward = 0
    if parser.iql_agent == 0:
        agent = INACAgent(parser.max_k, parser.alpha_0, parser.eta_0, parser.mode)
        for s in range(1, parser.max_step + 1):
            action = agent.step(reward, state)
            reward, state = env.step(action)
            episode_reward += sum(reward)
            if agent.k == 0 and s > 1:
                reward_list.append(episode_reward)
                episode_reward = 0
                state, reward = env.initialize()
    state, reward = env.initialize()
    episode_reward = 0
    for s in range(1, parser.test_step + 1):
        action = agent.step(reward, state, is_test=True)
        reward, state = env.step(action)
        episode_reward += sum(reward)
        if s % parser.max_k == 0:
            reward_list.append(episode_reward)
            episode_reward = 0
    return reward_list


def main():
    parser = get_parser()
    parser.iql_agent = 0
    reward_list = [[] for i in range(3)]
    for mode in [0, 1, 2]:
        parser.mode = mode
        for r in range(parser.num_repeat):
            reward = run(parser)
            reward_list[mode].append(reward)
    reward_ave = []
    reward_std = []
    for item in reward_list:
        reward_ = np.array(item)
        reward_ave.append(reward_.mean(axis=0) / (200 + 50))
        reward_std.append(reward_.std(axis=0) / (200 + 50))
    parser.iql_agent = 1
    iql_reward_list = [[] for i in range(3)]
    for mode in [0, 1, 2]:
        parser.mode = mode
        for r in range(parser.num_repeat):
            reward = run(parser)
            iql_reward_list[mode].append(reward)
    iql_reward_ave = []
    iql_reward_std = []
    for item in iql_reward_list:
        reward_ = np.array(item)
        iql_reward_ave.append(reward_.mean(axis=0) / (200 + 50))
        iql_reward_std.append(reward_.std(axis=0) / (200 + 50))
    n = len(reward_ave[0])
    x = [r for r in range(1, n + 1)]
    markers = ['-', '--', '-.']
    colors = ['blue', 'red', 'green']
    facecolors = ['skyblue', 'pink', 'yellowgreen']
    labels = ['Option 2 ', 'Option 3', 'Option 1']
    plt.rcParams['figure.figsize'] = (10, 4)
    fig = plt.subplot(121)
    for mode in [2, 0, 1]:
        print(len(x), len(iql_reward_ave[mode] - iql_reward_std[mode]), len(iql_reward_std[mode]))
        plt.plot(x, iql_reward_ave[mode], linestyle=markers[mode], color=colors[mode], linewidth=2, label=labels[mode])
        plt.fill_between(x, iql_reward_ave[mode] - iql_reward_std[mode], iql_reward_ave[mode] + iql_reward_std[mode],
                         where=iql_reward_ave[mode] - iql_reward_std[mode] <= iql_reward_ave[mode] + iql_reward_std[
                             mode],
                         facecolor=facecolors[mode])
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Normalized reward')
    plt.title('Performance of IQL')
    fig = plt.subplot(122)
    for mode in [2, 0, 1]:
        print(len(x), len(reward_ave[mode] - reward_std[mode]), len(reward_std[mode]))
        plt.plot(x, reward_ave[mode], linestyle=markers[mode], color=colors[mode], linewidth=2, label=labels[mode])
        plt.fill_between(x, reward_ave[mode] - reward_std[mode], reward_ave[mode] + reward_std[mode],
                         where=reward_ave[mode] - reward_std[mode] <= reward_ave[mode] + reward_std[mode],
                         facecolor=facecolors[mode])
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Normalized reward')
    plt.title('Performance of INAC')
    plt.savefig('./graph/artificial_case_compare.png', dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
