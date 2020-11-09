from regretmatching.rps import *
import numpy as np

print("=== rock win 100 , loose 1 ===")
RPS_REWARD_ROCK_Win100Loose1 = {
    ROCK:     np.asarray([0, 1, -100]),  # opponent playing ROCK
    PAPER:    np.asarray([-1, 0, 1]),  # opponent playing PAPER
    SCISSORS: np.asarray([100, -1, 0]),  # opponent playing SCISSORS
}
# test  win ratio
print("======== test : should more near Nash Equilibrium, more chance to win ")


a, b = calc_rps_nash(iteration=1000000, reward_table=RPS_REWARD_ROCK_Win100Loose1)

a.policy = a.average_policy()
t = 1000000

b.policy = np.full(3, 1. / 3)
a_win = 0
for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    a_win += a.reward_table[b_move][RPS_ACTIONS.index(a_move)]
print("vs 1/3,  a win ", a_win)


b.policy = np.array([1, 0, 0])
a_win = 0
for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    a_win += a.reward_table[b_move][RPS_ACTIONS.index(a_move)]
print("vs (1,0,0),  a win ", a_win)


b.policy = np.array([0, 1, 0])
a_win = 0
for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    a_win += a.reward_table[b_move][RPS_ACTIONS.index(a_move)]
print("vs (0,1,0),  a win ", a_win)


b.policy = np.array([0, 0, 1])
a_win = 0
for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    a_win += a.reward_table[b_move][RPS_ACTIONS.index(a_move)]
print("vs (0,0,1),  a win ", a_win)

# ----2.5902399604693143 - nash equilibrium for RPS: [0.01032854 0.9807096  0.00896186], [0.01014001 0.98107705 0.00878294]
# current policy for RPS: [0. 1. 0.], [0. 1. 0.]
# vs 1/3,  a win  39529
# vs (1,0,0),  a win  91514
# vs (0,1,0),  a win  -1208
# vs (0,0,1),  a win  45169
