from regretmatching.rps import *
import numpy as np


print("=== rock win 2 ,loose 2 ===")
#  锤子可以得2分，输2分
RPS_REWARD_ROCK_2 = {
    ROCK:     np.asarray([0, 2, -2]),  # opponent playing ROCK
    PAPER:    np.asarray([-2, 0, 1]),  # opponent playing PAPER
    SCISSORS: np.asarray([2, -1, 0]),  # opponent playing SCISSORS
}
calc_rps_nash(10000, reward_table=RPS_REWARD_ROCK_2)


print("=== rock win 2 , loose 1 ===")
RPS_REWARD_ROCK_W2L1 = {
    ROCK:     np.asarray([0, 1, -2]),  # opponent playing ROCK
    PAPER:    np.asarray([-1, 0, 1]),  # opponent playing PAPER
    SCISSORS: np.asarray([2, -1, 0]),  # opponent playing SCISSORS
}
calc_rps_nash(10000, reward_table=RPS_REWARD_ROCK_W2L1)

