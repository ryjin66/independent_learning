"""
files to help return tree-structured information
"""
import numpy as np


def tree_construction(agent_num=3):
    node = []
    a_node = []
    l_node = []
    depth_agent = {}
    agent_child = {}
    l_para = {}
    budget_range = []
    # 3 agents
    if agent_num == 3:
        node = [1, 2, 3, 4, 5, 6, 7]
        a_node = [1, 2, 3]
        l_node = [4, 5, 6, 7]
        para = [[10, 4, 4], [8, 4, 3], [7, 3, 2], [5, 2, 1], [6, 2, 1], [9, 3, 2], [50, 15, 5], [8, 4, 1], [50, 12, 5]]
        l_para = {4: [50, 15, 5, 0.5], 5: [8, 4, 3, 0.3], 6: [7, 3, 2, 0.5], 7: [5, 2, 1, 0.1]}
        budget_range = [4, 5]
        depth_agent = {1: [1], 2: [2, 3]}  # depth: agent
        agent_child = {1: (2, 3), 2: (4, 5), 3: (6, 7)}
    #  15 agents
    if agent_num == 15:
        node = [i for i in range(1, 32)]
        a_node = [i for i in range(1, 16)]
        l_node = [i for i in range(16, 32)]
        l_para = {16: [5, 2, 1, 0.5], 17: [8, 4, 3, 0.5], 18: [8, 4, 3, 0.5], 19: [10, 4, 4, 0.5], 20: [7, 3, 2, 0.5],
                  21: [8, 4, 3, 0.5], 22: [6, 2, 1, 0.5], 23: [50, 12, 5, 0.5], 24: [50, 15, 5, 0.5], 25: [6, 2, 1, 0.5],
                  26: [7, 3, 2, 0.5], 27: [8, 4, 1, 0.5], 28: [50, 12, 5, 0.5], 29: [8, 4, 1, 0.5], 30: [6, 2, 1, 0.5], 31: [7, 3, 2, 0.5]}
        depth_agent = {1: [1], 2: [2, 3], 3: [4, 5, 6, 7], 4: [8, 9, 10, 11, 12, 13, 14, 15]}  # depth: agent
        agent_child = {1: (2, 3), 2: (4, 5), 3: (6, 7), 4: [8, 9], 5: [10, 11], 6: [12, 13], 7: [14, 15], 8: [16, 17],
                       9: [18, 19], 10: [20, 21], 11: [22, 23], 12: [24, 25], 13: [26, 27], 14: [28, 29], 15: [30, 31]}
        budget_range = [15, 17]
    return node, a_node, l_node, l_para, depth_agent, agent_child, budget_range


def information_construct(depth_agent_, agent_child_, information_, contraction):
    """
    Construct the contracted information for each agent
    """
    depth = max(depth_agent_.keys())
    for layer in range(depth, 0, -1):
        for node in depth_agent_[layer]:
            child = agent_child_[node]
            inform1 = information_[child[0]]
            inform2 = information_[child[1]]
            if contraction == 'Full information':
                inform_node = np.r_[inform1, inform2]
            if contraction == 'Averaged information':
                if inform1[1] + inform2[1] == 0:
                    e1 = 0.5
                else:
                    e1 = inform1[1] / (inform1[1] + inform2[1])
                inform_node = e1 * inform1 + (1 - e1) * inform2
                inform_node[1] = inform1[1] + inform2[1]
            if contraction == 'No information':
                inform_node = 0.0 * inform1 + 0.0 * inform2
            information_[node] = inform_node
    return information_


def reward_construct(depth_agent_, agent_child_, reward_):
    depth = max(depth_agent_.keys())
    for layer in range(depth, 0, -1):
        for node in depth_agent_[layer]:
            child = agent_child_[node]
            reward1 = reward_[child[0]]
            reward2 = reward_[child[1]]
            reward_[node] = reward1 + reward2
    return reward_

