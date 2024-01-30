"""
Train NPG-based algorithm
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from evcn_envs import evcn
import main_tool as mt
from NPG_agent import *
import baseline_run

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OPTIMAL_DIC = {3: 102.46, 15: 511.62}
DATA_SAVE_PATH = './data/NPG_with_diff_contractions'


def sigmoid(x):
    if x < -pow(10, 1):
        z = 0
    elif x > pow(10, 1):
        z = 1
    else:
        z = 1 / (1 + np.exp(-x))
    return z


def state_action_construct(depth_agent_, node_agent_, agent_child_, total_budget_, information_, is_test_):
    budget_ = {}
    action_ = {}
    state_ = {}
    root = depth_agent_[1][0]
    budget_[root] = total_budget_
    depth = max(depth_agent_.keys())
    for layer in range(1, depth + 1):
        for node in depth_agent_[layer]:
            child = agent_child_[node]
            inform1 = information_[child[0]]
            inform2 = information_[child[1]]
            agent = node_agent_[node]
            node_state = np.r_[budget_[node] / 10, inform1, inform2]
            action_[node] = agent.action_step(node_state, is_test_)
            state_[node] = node_state
            budget_[child[0]] = float(sigmoid(action_[node]) * budget_[node])
            budget_[child[1]] = budget_[node] - budget_[child[0]]
    return state_, action_, budget_


def run_npg(parser, method):
    env = evcn(parser.agent_num)
    depth = max(env.depth_agent.keys())
    agent_child = env.agent_child
    agent_node = env.a_node
    leaves_node = env.l_node
    depth_agent = env.depth_agent
    node_agent = {}
    if method == 'baseline':
        base_reward_list = []
        for run in range(parser.run_time):
            base_reward = baseline_run.baseline_main(parser, env)
            base_reward_list.append(base_reward[1])
        base_rew = np.array(base_reward_list)
        base_ave_reward = base_rew.mean(axis=0)
        base_std_reward = base_rew.std(axis=0)
        return base_ave_reward, base_std_reward
    total_reward_list = []
    for run in range(parser.run_time):
        for layer in range(1, depth + 1):
            for node in depth_agent[layer]:
                node_agent[node] = agent(parser.state_dim, parser.action_dim, parser.sigma, parser.gamma,
                                         parser.p_learning_rate * pow(10, layer - 1),
                                         parser.q_learning_rate * pow(10, layer - 1),
                                         parser.buffer_size, parser.batch_size)
        if method == 'Full information':
            for node in depth_agent[1]:
                node_agent[node] = agent(21, parser.action_dim, parser.sigma, parser.gamma,
                                         parser.p_learning_rate * pow(10, 0), parser.q_learning_rate * pow(10, 0),
                                         parser.buffer_size, parser.batch_size)
        information, total_budget = env.initialize()
        i = 0
        last_state = 0
        last_reward = 0
        last_action = 0
        total_reward = []
        reward_list = {node: 0 for node in agent_node}
        is_test = False
        while True:
            i += 1
            # construct the information of each agent
            information = mt.information_construct(depth_agent, agent_child, information, contraction=method)

            # construct state and action of each agent
            state, action, budget = state_action_construct(depth_agent, node_agent, agent_child, total_budget,
                                                           information, is_test)
            # store the sample for each agent
            if i > 1:
                for node in agent_node:
                    node_agent[node].replay_buffer.push(
                        [last_state[node], last_action[node], last_reward[node], state[node]])
            # update
            if i % 5 == 0 and not is_test:
                for node in agent_node:
                    node_agent[node].critic_update()
            if i % 48 == 0 and not is_test:
                for node in agent_node:
                    node_agent[node].policy_update()
            for node in agent_node:
                node_agent[node].actor.sigma = max(
                    (parser.sigma - i * (parser.sigma - parser.min_sigma) / (parser.max_steps * 0.75)), parser.min_sigma)
            if i == parser.max_steps / 2:
                for node in agent_node:
                    node_agent[node].actor.sigma = parser.min_sigma
            if i >= 0.75 * parser.max_steps:
                is_test = True
            # execute actions to the env
            budget_leaves = {node: budget[node] for node in leaves_node}
            information, total_budget, reward = env.step(budget_leaves)
            # log states, actions and rewards
            last_state = state
            last_action = action
            last_reward = mt.reward_construct(depth_agent, agent_child, reward)
            for node in agent_node:
                reward_list[node] = reward_list[node] + last_reward[node]
            if i > 0 and i % 240 == 0:
                total_reward.append(reward_list[1])
                reward_list[1] = 0
            if i == parser.max_steps:
                break

        total_reward_list.append(total_reward)

    rew = np.array(total_reward_list)
    ave_reward = rew.mean(axis=0)
    std_reward = rew.std(axis=0)

    return ave_reward, std_reward


def main(parser):
    contraction_list = ['Full information', 'Averaged information', 'No information'] if parser.agent_num == 3 else [
        'Averaged information', 'No information']
    reward_c = []
    std_reward_c = []
    for contraction in contraction_list:
        rew_, std_ = run_npg(parser, contraction)
        reward_c.append(rew_)
        std_reward_c.append(std_)
    base_rew, base_std = run_npg(parser, 'baseline')
    # ----SAVE RESULTS----
    if parser.agent_num == 3:
        result = {'reward1': reward_c[0], 'std_1': std_reward_c[0],
                  'reward2': reward_c[1], 'std_2': std_reward_c[1],
                  'reward3': reward_c[2], 'std_3': std_reward_c[2],
                  'reward_base': base_rew, 'std_base': base_std}
    else:
        result = {'reward1': reward_c[0], 'std_1': std_reward_c[0],
                  'reward2': reward_c[1], 'std_2': std_reward_c[1],
                  'reward_base': base_rew, 'std_base': base_std}
    DATA_SAVE_PATH = './data/NPG_with_diff_contractions_{}_agent_{}.csv'.format(parser.agent_num, parser.case_num)
    result_df = pd.DataFrame(result).to_csv(DATA_SAVE_PATH, index=False)
    # ----PLOT----
    markers = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1)), (0, (1, 2, 3, 4, 2, 2))]
    colors = ['blue', 'red', 'green', 'purple']
    facecolors = ['skyblue', 'pink', 'yellowgreen', 'mediumpurple']
    plt.rcParams['figure.figsize'] = (6.0, 5.0)
    x = [i for i in range(1, len(base_rew) + 1)]
    # contraction=['Full information','Averaged information','No information']
    for i in range(len(reward_c)):
        rew = reward_c[i] / OPTIMAL_DIC[parser.agent_num]
        std = std_reward_c[i] / OPTIMAL_DIC[parser.agent_num]
        plt.plot(x, rew, linestyle=markers[i], linewidth=1, color=colors[i], label=contraction_list[i])
        plt.fill_between(x, rew - std, rew + std,
                         where=rew - std <= rew + std, facecolor=facecolors[i])
    rew = base_rew / OPTIMAL_DIC[parser.agent_num]
    std = base_std / OPTIMAL_DIC[parser.agent_num]
    plt.plot(x, rew, linestyle=markers[3], linewidth=1, color=colors[3], label='Baseline')
    plt.fill_between(x, rew - std, rew + std,
                     where=rew - std <= rew + std, facecolor=facecolors[3])
    plt.ylabel("Normalized reward")
    plt.xlabel("Episode")
    plt.title("Performance of baseline and our algorithm")
    plt.legend()
    plt.grid(True)
    if parser.save_plot:
        plt.savefig('./graph/NPG_{}agents_normalized_reward_compare_{}.png'.format(parser.agent_num, parser.case_num),
                    dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    para = argparse.ArgumentParser()
    para.add_argument('--case_num', type=int, default=2)
    para.add_argument('--state_dim', type=int, default=11)
    para.add_argument('--action_dim', type=int, default=1)
    para.add_argument('--gamma', type=float, default=0.99)
    para.add_argument('--sigma', type=float, default=10)
    para.add_argument('--min_sigma', type=float, default=0.5)
    para.add_argument('--sigma_decay', type=float, default=0.7)
    para.add_argument('--p_learning_rate', type=float, default=0.00005)
    para.add_argument('--q_learning_rate', type=float, default=0.0001)
    para.add_argument('--buffer_size', type=int, default=3000)
    para.add_argument('--batch_size', type=int, default=32)
    para.add_argument('--max_steps', type=int, default=50000)
    para.add_argument('--agent_num', type=int, default=3)
    para.add_argument('--run_time', type=int, default=10)
    para.add_argument('--save_plot', type=bool, default=True)
    para.add_argument('--log_info', type=bool, default=False)
    parser = para.parse_args()
    main(parser)
