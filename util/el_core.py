import time, math
from numba import jit
from math import lgamma
import scipy.stats as stats
import numpy as np
import util.helper as helper

@jit
def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return math.exp(num - den)

@jit
def g0(a, b, c):    
    return math.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))

@jit
def hiter(a, b, c, d):
    total = 0.0
    for i in range(1, a):
        total += h(i, b, c, d) / i
    return total

@jit
def calc_prob(a, b, c, d):
    return g0(d, b, c) + hiter(a, b, c, d)

@jit
def calc_beta(a, b):   
    return  math.exp(lgamma(a+1) - lgamma(a+b+1) - lgamma(a) + lgamma(a+b))

@jit
def calc_loss(a, b, c, d):
    x1 = calc_beta(a, b)*calc_prob(a+1, b, c, d)
    x2 = calc_beta(c, d)*calc_prob(a, b, c+1, d)
    return x1 - x2


def calc_stat(sa, na, sb, nb, relative_mde_value=0.02, toc_adj_factor=0.00025, ci_prob=0.99):

    alpha_a = sa + 1
    beta_a = 1 + na - sa
    cvr_a = alpha_a/(alpha_a+beta_a)
    x = (1-ci_prob)/2
    cvr_a_lb = stats.beta.ppf(x, alpha_a, beta_a)
    cvr_a_ub = stats.beta.ppf(1-x, alpha_a, beta_a)
    
    alpha_b = sb + 1
    beta_b = 1 + nb - sb
    cvr_b = alpha_b/(alpha_b+beta_b)
    cvr_b_lb = stats.beta.ppf(x, alpha_b, beta_b)
    cvr_b_ub = stats.beta.ppf(1-x, alpha_b, beta_b)
    lift = cvr_b - cvr_a 
    
    prob_a_gt_b = calc_prob(alpha_a, beta_a, alpha_b, beta_b)
    prob_b_gt_a = 1 - prob_a_gt_b

    # compute expected loss(el)
    el_a = calc_loss(alpha_b, beta_b, alpha_a, beta_a)
    el_b = calc_loss(alpha_a, beta_a, alpha_b, beta_b)

    ret = helper.estimate_toc(mu=cvr_a, relative_mde_value=relative_mde_value,
                              toc_adj_factor=toc_adj_factor, pr=False)
    toc = ret['toc']

    if el_a < toc and el_b < toc:
        el_res = 'E'
    elif el_a < toc and el_b > toc:
        el_res = 'A'
    elif el_a > toc and el_b < toc:
        el_res = 'B'
    else:
        el_res = 'U'

    return {'data': [sa, na, sb, nb],
            'beta_param': [alpha_a, beta_a, alpha_b, beta_b],
            'cvr_a': cvr_a, 
            'cvr_b': cvr_b, 
            'cvr_a_ci': [cvr_a_lb, cvr_a_ub],
            'cvr_b_ci': [cvr_b_lb, cvr_b_ub],
            'prob_b_gt_a': prob_b_gt_a,
            'lift': lift,
            'el_a': el_a, 
            'el_b': el_b,
            'el_res': el_res}


def calc_ab(data, relative_mde_value=0.02, toc_adj_factor=0.00025, ci_prob=0.99):

    sa = np.sum(data[0])
    na = len(data[0])

    sb = np.sum(data[1])
    nb = len(data[1])

    return calc_stat(sa, na, sb, nb, relative_mde_value, toc_adj_factor, ci_prob)
