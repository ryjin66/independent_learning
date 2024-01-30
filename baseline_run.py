"""
Run the baseline algorithm
"""
import main_tool as mt


def information_construct(depth_agent_, agent_child_, information_):
    # construct the contracted information for each agent
    depth = max(depth_agent_.keys())
    for layer in range(depth, 0, -1):
        for node in depth_agent_[layer]:
            child = agent_child_[node]
            inform1 = information_[child[0]]
            inform2 = information_[child[1]]
            if inform1[1] + inform2[1] == 0:
                e1 = 0.5
            else:
                e1 = inform1[1] / (inform1[1] + inform2[1])
            inform_node = e1 * inform1 + (1 - e1) * inform2
            inform_node[1] = inform1[1] + inform2[1]
            if inform_node[1] > 0:
                inform_node[4] = inform1[4] + inform2[4]
            information_[node] = inform_node
    return information_


def state_action_construct(depth_agent_, agent_child_, total_budget_, information_):
    """
    Construct actions for baseline policy
    """
    budget_ = {}
    action_ = {}
    state_ = {}
    root = depth_agent_[1][0]
    budget_[root] = total_budget_
    depth = max(depth_agent_.keys())
    for layer in range(1, depth + 1):
        for node in depth_agent_[layer]:
            child = agent_child_[node]
            # Baseline policy: P_rate
            inform1 = information_[child[0]]
            inform2 = information_[child[1]]
            if (inform1 + inform2)[1] == 0:
                e1 = 0.5
            else:
                e1 = inform1[1] / (inform1[1] + inform2[1])
            action_[node] = e1
            budget_[child[0]] = action_[node] * budget_[node]
            budget_[child[1]] = budget_[node] - budget_[child[0]]
    return state_, action_, budget_


def baseline_main(para, env):
    parser = para
    information, total_budget = env.initialize()
    depth = max(env.depth_agent.keys())
    agent_child = env.agent_child
    agent_node = env.a_node
    leaves_node = env.l_node
    depth_agent = env.depth_agent
    leaf_pile = env.leaf_pile
    i = 0

    reward_dic = {node: [] for node in agent_node}
    reward_list = {node: 0 for node in agent_node}
    while True:
        i += 1
        # construct the information of each agent
        information = information_construct(depth_agent, agent_child, information)

        # construct state and action of each agent
        state, action, budget = state_action_construct(depth_agent, agent_child, total_budget, information)

        # execute actions to the env
        budget_leaves = {node: budget[node] for node in leaves_node}
        information, total_budget, reward = env.step(budget_leaves)
        # reward construction
        last_reward = mt.reward_construct(depth_agent, agent_child, reward)
        for node in agent_node:
            reward_list[node] = reward_list[node] + last_reward[node]
        if i > 0 and i % 240 == 0:
            for node in agent_node:
                reward_dic[node].append(reward_list[node])
                reward_list[node] = 0
        if i == parser.max_steps:
            break
    return reward_dic
