import sys, time
from datetime import timedelta
import numpy as np
import pandas as pd
import scipy.stats as stats
import util.helper as helper
import util.msprt_core as msprt_core


def sim_peeking(muA, muB, sample_size, n_experiments, n_peeks=-1, start=None, 
                alpha=0.05, relative_mde_value=0.02, 
                tau_option=3, tau_constant=0.0001, burnIn=0, 
                random_seed=-1, pr_peek=1, pr_fixed=False, fn=None):

    if random_seed > 0:
        np.random.seed(random_seed)

    start_time = time.time()
    helper.fprint("[Start time]:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

    ss = helper.sample_size(muA, relative_mde_value=relative_mde_value)
    freq_sample_size = ss['variant_sample_size']

    if isinstance(start, float):
        assert (0 < start and start < 1)
        start = int(freq_sample_size*start)
        if n_peeks == -1:
            step = 1
        else:
            step = int((sample_size-start)/(n_peeks-1))
    elif isinstance(start, int):
        assert (1 <= start and start <= sample_size)
        if n_peeks == -1:
            step = 1
        else:
            step = int((sample_size-start)/(n_peeks-1))
    else:
        if n_peeks == -1:
            start = 1
            step = 1
        else:
            start = int(sample_size/n_peeks)
            step = start

    absolute_mde_value = muA*relative_mde_value

    # print parameter settings
    helper.fprint("[Parameters]: muA:{}, muB:{}, sample_size:{:,}, n_experiments:{:,},"
                  " relative_mde_value:{:.5f}, alpha:{:.3f}, tau_option:{}, tau_constant:{},"
                  " burnIn:{:,}, random_seed:{}, n_peeks:{:,}, start:{:,}, step: {:,}"
                  .format(muA, muB, sample_size, n_experiments, 
                          relative_mde_value, alpha, tau_option, tau_constant, 
                          burnIn, random_seed, n_peeks, start, step), fn)

    sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments)

    sim_res_data = []
    msprt_res = ['U']*n_experiments
    msprt_ss = [0]*n_experiments # record the sample size when concluded
    msprt_effect = [0]*n_experiments
    msprt_prev_p_val = [1]*n_experiments
    msprt_prev_ci = [[-1000,1000]]*n_experiments
    peek_idx = 0
    for n in range(start, sample_size+1, step):  
        peek_idx += 1
        peek_sig_effect = []
        peek_effect = []
        peek_decision_boundary = []
        for s in range(n_experiments):
            if msprt_res[s] == 'U':
                sa = sim_data[s][0][n-1]
                sb = sim_data[s][1][n-1]
                res = msprt_core.calc_stat(sa, n, sb, n, alpha,
                                           msprt_prev_p_val[s], msprt_prev_ci[s])
                msprt_res[s] = res['msprt_res']
                msprt_ss[s] = n
                msprt_effect[s] = res['effect']
                msprt_prev_p_val[s] = res['p_value']
                msprt_prev_ci[s] = res['effect_ci']
                if res['msprt_res'] != 'U':
                    peek_sig_effect.append(res['effect'])
                peek_effect.append(res['effect'])
                peek_decision_boundary.append(res['decision_boundary'])

        res = pd.Series(msprt_res).value_counts().to_dict()
        if len(peek_sig_effect) == 0:
            peek_avg_sig_effect = 0
        else:
            peek_avg_sig_effect = np.mean(abs(np.array(peek_sig_effect)))

        if len(peek_effect) == 0:
            peek_avg_effect = 0
        else:
            peek_avg_effect = np.mean(abs(np.array(peek_effect)))

        if len(peek_decision_boundary) == 0:
            peek_avg_decision_boundary = 0
        else:
            peek_avg_decision_boundary = np.mean(abs(np.array(peek_decision_boundary)))

        sim_res_data_dict = {}
        sim_res_data_dict.update(res)
        sim_res_data_dict['peek_id'] = peek_idx
        sim_res_data_dict['samples'] = n 
        sim_res_data_dict['decision_boundary'] = peek_avg_decision_boundary
        sim_res_data_dict['avg_sig_effect'] = peek_avg_sig_effect
        sim_res_data_dict['avg_effect'] = peek_avg_effect
        sim_res_data.append(sim_res_data_dict)
            
        # peek summary
        if pr_peek > 0 and peek_idx%pr_peek == 0:
            helper.fprint("[{}]: Peek #{:,} @ {:,} samples,"
                          " avg_sig_effect:{:.5f},"
                          " avg_effect:{:.5f}, msprt_res:{}"
                         .format(time.strftime("%m-%d %H:%M:%S", time.localtime()), peek_idx, n, 
                                 peek_avg_sig_effect,
                                 peek_avg_effect, res), fn)

        if 'U' not in msprt_res:
            break

    # print parameters settings
    helper.fprint("[Parameters]: muA:{}, muB:{}, sample_size:{:,}, n_experiments:{:,},"
                  " relative_mde_value:{:.5f}, alpha:{:.3f}, tau_option:{}, tau_constant:{}," 
                  " burnIn:{:,}, random_seed:{}, n_peeks:{:,}, start:{:,}, step:{:,}"
                  .format(muA, muB, sample_size, n_experiments, 
                          relative_mde_value, alpha, tau_option, tau_constant,
                          burnIn, random_seed, n_peeks, start, step), fn)

    # print simulation summary
    winner_ss = []
    winner_effect = []
    if muA == muB:

        for i, r in enumerate(msprt_res):
            if r in (['A','B']):
                winner_effect.append(msprt_effect[i])

        helper.fprint("[mSPRT][{:,} peeks][{:,} samples] - Type 1 error rate:{:.4f},"
                      " error_avg_effect:{:.5f}, msprt_res:{}"
                      .format(peek_idx, n, 1-(res.get('U',0)+res.get('E',0))/n_experiments, 
                              np.mean(abs(np.array(winner_effect))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(msprt_res):
            if r == winner:
                winner_ss.append(msprt_ss[i])
                winner_effect.append(msprt_effect[i])

        helper.fprint("[mSPRT][{:,} peeks][{:,} samples] - Power:{:.4f}, avg_sample_size:{:,.0f},"
                      " sample_size_ratio:{:.2f}%, avg_effect:{:.5f}, msprt_res:{}"
                      .format(peek_idx, n, res.get(winner,0)/n_experiments, np.mean(winner_ss), 
                              np.mean(winner_ss)/freq_sample_size*100, 
                              np.mean(abs(np.array(winner_effect))), res), fn)

    # show fixed sample size result     
    if pr_fixed:
        sim_fixed(muA, muB, sample_size, n_experiments, sim_data, relative_mde_value, alpha)

    helper.fprint("[Elasped time]:{}".format(str(timedelta(seconds=time.time()-start_time))), fn)
    helper.fprint("[End time]:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

    return {'sim_res_df': pd.DataFrame(sim_res_data), 
            'msprt_res': msprt_res,
            'msprt_effect': msprt_effect}


def sim_fixed(muA, muB, sample_size, n_experiments, sim_data=None, relative_mde_value=0.02, alpha=0.05):

    if sim_data is None:
        sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments)

    msprt_res = ['U']*n_experiments
    msprt_effect = [0]*n_experiments
    for s in range(n_experiments):
        sa = sim_data[s][0][sample_size-1]
        sb = sim_data[s][1][sample_size-1]
        res = msprt_core.calc_stat(sa, sample_size, sb, sample_size, relative_mde_value, alpha, 
                                   tau_option, tau_constant) 
        msprt_res[s] = res['msprt_res']
        msprt_effect[s] = res['effect']

    res = pd.Series(msprt_res).value_counts().to_dict()

    absolute_mde_value = muA*relative_mde_value
    min_msprt_half_width = absolute_mde_value*msprt_mde_ratio

    helper.fprint("[Parameters]: muA:{}, muB:{}, sample_size:{:,}, n_experiments:{:,},"
                  " min_msprt_half_width:{:.6f}"
                  .format(muA, muB, sample_size, n_experiments, relative_mde_value), fn)

    winner_effect = []
    if muA == muB:

        for i, r in enumerate(msprt_res):
            if r in (['A','B']):
                winner_effect.append(msprt_effect[i])

        helper.fprint("[mSPRT] Fixed - Type 1 error rate:{:.4f}, error_avg_effect:{:.5f}, msprt_res:{}"
                      .format(1-(res.get('U',0)+res.get('E',0))/n_experiments,
                              np.mean(abs(np.array(winner_effect))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(msprt_res):
            if r == winner:
                winner_effect.append(msprt_effect[i])

        helper.fprint("[mSPRT] Fixed - Power:{:.4f}, avg_effect:{:.5f}, msprt_res:{}"
                      .format(res.get(winner,0)/n_experiments, np.mean(abs(np.array(winner_effect))), res), fn)

    return {'msprt_res': msprt_res, 
            'msprt_effect': msprt_effect}
