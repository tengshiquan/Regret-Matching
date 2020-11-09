from regretmatching.model import RegretMatchingDecisionMaker, Action
import numpy as np

ROCK = Action('rock')
PAPER = Action('paper')
SCISSORS = Action('scissors')

RPS_ACTIONS = [ROCK, PAPER, SCISSORS]

RPS_REWARD_VECTORS = {
    ROCK:     np.asarray([0, 1, -1]),  # opponent playing ROCK
    PAPER:    np.asarray([-1, 0, 1]),  # opponent playing PAPER
    SCISSORS: np.asarray([1, -1, 0]),  # opponent playing PAPER
}


class RPSPlayer(RegretMatchingDecisionMaker):
    def __init__(self, reward_table=RPS_REWARD_VECTORS):
        super(RPSPlayer, self).__init__(RPS_ACTIONS, reward_table)

    def move(self):
        return self.decision()


def calc_rps_nash(iteration=10000, RMplus=False, reward_table=RPS_REWARD_VECTORS):
    a = RPSPlayer(reward_table)
    b = RPSPlayer(reward_table)
    a.RM_plus = RMplus
    b.RM_plus = RMplus

    for i in range(0, iteration):
        a.learn_from(b.move())
        b.learn_from(a.move())

    nash = np.full(3, 1. / 3)
    print("----{0} - nash equilibrium for RPS: {1}, {2}".format(a.dis_from_nash(nash)+b.dis_from_nash(nash),
                                                                a.average_policy(), b.average_policy()))
    print("current policy for RPS: {0}, {1}".format(a.policy, b.policy))

    return a, b