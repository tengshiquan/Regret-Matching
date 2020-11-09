from regretmatching.rps import *

if __name__ == '__main__':
    print("====== Regret Matching =====")
    calc_rps_nash(RMplus=False)
    print("======RM plus should converge faster=====")
    calc_rps_nash(RMplus=True)







