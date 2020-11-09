import numpy as np


class Action:
    def __init__(self, id):
        self.id = id


class RegretMatchingDecisionMaker:

    def __init__(self, actions, reward_table):
        self.n = len(actions)
        # action list
        self.actions = actions

        self.reward_table = reward_table

        # cumulative regrets towards experts
        self.regrets = np.zeros(self.n)

        self.policy = np.full(self.n, 1. / self.n)

        self.sum_p = np.full(3, 0.)

        self.RM_plus = False

    def decision(self):
        action = np.random.choice(self.actions, 1, p=self.policy)
        return action[0]

    def update_rule(self, rewards_vector):  # rewards_vector 是各个action的收益
        expected_reward = np.dot(self.policy, rewards_vector)  # 当前策略预期收益，是个标量
        self.regrets += (rewards_vector - expected_reward)

        if self.RM_plus:
            self.regrets[self.regrets < 0] = 0

        self._update_p()

    def _update_p(self):  # R+(A) / Sum(R+(a))
        sum_w = np.sum([self._w(i) for i in np.arange(self.n)])  #
        if sum_w <= 0:
            self.policy = np.full(self.n, 1. / self.n)
        else:
            self.policy = np.asarray(
                [self._w(i) / sum_w for i in np.arange(self.n)]
            )

        self.sum_p += self.policy  # 直接相加，再除 T 就是 平均策略

    def _w(self, i):
        return max(0, self.regrets[i])

    def learn_from(self, opponent_move):  # 根据对手当前的move ，求regret ，来update
        reward_vector = self.reward_table[opponent_move]
        self.update_rule(reward_vector)

    def average_policy(self):
        return self.sum_p / sum(self.sum_p)

    def dis_from_nash(self, nash):
        return sum(abs(self.average_policy()-nash))

    def eps(self):
        return np.max(self.regrets / sum(self.sum_p))