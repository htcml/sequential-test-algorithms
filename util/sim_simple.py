import time, math
from math import lgamma
from numba import jit
import numpy as np
import pandas as pd
import scipy.stats as stats


def calc_ab(data, n):
    c = np.sum(data[0])
    t = np.sum(data[1])

    if t - c > 2*math.pow(n, 0.5):
        winner = 'B'
    elif c - t > 2*math.pow(n, 0.5):
        winner = 'A'
    elif t + c > n:
        winner = 'S'
    else:
        winner = 'U'

    #print(c, t, t-c, int(2*math.pow(n, 0.5)), winner)

    return winner

def sim_peeking(data, n, start, step, pr=True):

    pairs = len(data)
    seq_len = len(data[0][0])

    ti_res = ['U']*pairs
    ti_len = [0]*pairs

    peek_cnt = 0
    for l in range(start, seq_len+1, step):  
        peek_cnt += 1
        for j in range(pairs):

            if ti_res[j] == 'U':

                data1 = data[j][0][:l]
                data2 = data[j][1][:l]
                ab_res = calc_ab([data1, data2], n) 
                ti_res[j] = ab_res
                ti_len = l

            if ab_res == 'S':
                break

        if ('U' not in ti_res):
            break
            
        if pr:
            print("peek #{} at first {} data points, TI_res: {}"
                  .format(peek_cnt, l, pd.Series(ti_res).value_counts().to_dict()))

    return pd.Series(ti_res).value_counts().to_dict(), ti_len

def sim_fixed(data, n):

    pairs = len(data)
    ti_res = ['U']*pairs

    for j in range(pairs):
        data1 = data[j][0]
        data2 = data[j][1]
        ab_res = calc_ab([data1, data2], n)
        ti_res[j] = ab_res
                
    return pd.Series(ti_res).value_counts().to_dict()
                

