"""
Environment files
"""
import numpy as np
from main_tool import tree_construction


class evcn():
    def __init__(self,agent_num):
        self.agent_child = {}
        self.depth_agent = {}
        self.node = []
        self.a_node = {}
        self.l_node = []
        self.leaf_pile = {}
        self.l_para = {}
        self.total_budget = 0
        self.agent_naive = {}
        self.budget_range = []
        self.log = {'cp_node': [], 'arrival': [], 'departure': [], 'demand': [], 'P_rate': []}
        self.budget_list = {'budget': []}
        self.tree_cons(agent_num)

    def tree_cons(self, agent_num):
        self.node, self.a_node, self.l_node, self.l_para, self.depth_agent, self.agent_child, self.budget_range = tree_construction(agent_num)
        self.leaf_pile = {node: charging_pile(node, self.l_para[node]) for node in self.l_node}  # leaf: charging pile

    def initialize(self):
        information = {}
        for node in self.l_node:
            pile = self.leaf_pile[node]
            if pile.is_connected == 1 and pile.full_charged == 0:
                information[node] = np.array([pile.d, pile.P_rate/10, pile.ft/24, pile.lt/24, pile.demand/50])
            else:
                information[node] = np.array([0, 0, 0, 0, 0])
        self.total_budget = np.random.uniform(self.budget_range[0], self.budget_range[1])
        return information, self.total_budget

    def step(self, action):
        # action is a dictionary: {agent: max_budget}
        information = {}
        reward = {}
        for node in self.l_node:
            pile = self.leaf_pile[node]
            pile.pile_step(action[node])
            if pile.is_connected == 1 and pile.full_charged == 0:
                information[node] = np.array([pile.d, pile.P_rate/10, pile.ft/24, pile.lt/24, pile.demand/50])
            else:
                information[node] = np.array([0, 0, 0, 0, 0])
            reward[node] = pile.pile_reward
        self.total_budget = 0.3 * self.total_budget + 0.7 * np.random.uniform(self.budget_range[0],
                                                                              self.budget_range[1])
        return information, self.total_budget, reward

    def information_log(self, i, max_steps):
        self.budget_list['budget'].append(self.total_budget)
        for node in self.l_node:
            pile = self.leaf_pile[node]
            if pile.lt > 0 and pile.lt == pile.stay_time:
                self.log['cp_node'].append(node)
                self.log['arrival'].append(i)
                self.log['departure'].append(min(np.ceil(i + pile.stay_time), max_steps))
                self.log['demand'].append(pile.demand)
                self.log['P_rate'].append(pile.P_rate)


class charging_pile():
    def __init__(self, node, EV_para):
        self.node = node
        self.is_connected = 0
        self.full_charged = 0
        self.demand = 0
        self.d = 0
        self.P_rate = 0
        self.ft = 0
        self.lt = 0
        self.pile_reward = 0
        self.generate_parameter = EV_para
        self.stay_time = 0
        self.initialize()

    def initialize(self):
        if self.is_connected == 0:
            # the parameter is arrival rate, they can be different for each charging pile
            self.generate_EV(self.generate_parameter[3])
            self.full_charged = 0
        else:
            # set the P_rate and ft, d= 0
            self.P_rate = 0
            self.ft = 0
            self.d = 0

    def pile_step(self, max_budget):
        if self.is_connected == 0:
            self.initialize()
            self.pile_reward = 0
        elif self.is_connected == 1 and self.full_charged == 1:
            self.lt = self.lt - 1
            self.pile_reward = 0
        else:
            P_charge = min(max_budget, self.P_rate)
            last_d = self.d + 0
            self.d = float(min(self.d + P_charge * 1 / self.demand, 1))
            self.pile_reward = self.d - last_d
            self.lt = self.lt - 1
            self.ft = self.demand * (1 - self.d) / self.P_rate
            if self.d == 1:
                self.initialize()
                self.full_charged = 1
        if self.lt <= 0:
            self.is_connected = 0
            self.initialize()

    def generate_EV(self, arrival_rate):
        number = np.random.poisson(arrival_rate)
        if self.generate_parameter[0] == 0:
            pass
        if number >= 1:
            self.is_connected = 1
            self.demand = np.random.uniform(self.generate_parameter[0] - 2,
                                            self.generate_parameter[0] + 2)
            self.d = 0
            self.P_rate = np.random.uniform(self.generate_parameter[1] - 0.5, self.generate_parameter[1] + 1)
            self.ft = np.ceil(self.demand / self.P_rate)
            self.lt = self.ft + np.random.uniform(self.generate_parameter[2] - 0.5, self.generate_parameter[2] + 1)
            self.stay_time = self.lt + 0
        else:
            self.d = 0
            self.P_rate = 0
            self.lt = 0
            self.ft = 0
