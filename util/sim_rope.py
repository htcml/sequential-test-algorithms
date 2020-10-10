import sys, time
from datetime import timedelta
import numpy as np
import pandas as pd
import scipy.stats as stats
import util.helper as helper
import util.rope_core as rope_core


def sim_peeking(muA, muB, sample_size, n_experiments, n_peeks=-1, start=None, 
                relative_mde_value=0.02,
                rope_hdi_ratio=0.6, rope_mde_ratio=0.375, 
                hdi_cred_mass=0.95, rvs_size=1000000, 
                pr_peeking=True, pr_fixed=False, fn=None):

    start_time = time.time()
    helper.fprint("[Start time]: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

    ss = helper.sample_size(muA, mde_value=relative_mde_value, relative_mde=True)
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
    min_rope_half_width = absolute_mde_value*rope_mde_ratio

    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value: {}, rope_hdi_ratio: {:.2f}, rope_mde_ratio: {:.3f},"
                  " min_rope_half_width: {:.5f}, n_peeks: {:,}, start: {:,}, step: {:,}, rvs_size: {:,}"
                  .format(muA, muB, sample_size, n_experiments, relative_mde_value, 
                          rope_hdi_ratio, rope_mde_ratio, min_rope_half_width, 
                          n_peeks, start, step, rvs_size), fn)

    sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments)

    sim_res_data = []
    rope_res = ['U']*n_experiments
    rope_ss = [0]*n_experiments # record the sample size when concluded
    rope_lift = [0]*n_experiments
    peek_idx = 0
    for n in range(start, sample_size+1, step):  
        peek_idx += 1
        peek_hdi_width = []
        peek_rope_width = []
        peek_lift = []
        for s in range(n_experiments):
            if rope_res[s] == 'U':
                sa = sim_data[s][0][n-1]
                sb = sim_data[s][1][n-1]
                res = rope_core.calc_stat(sa, n, sb, n, relative_mde_value, 
                                          rope_hdi_ratio, rope_mde_ratio, hdi_cred_mass, rvs_size) 
                rope_res[s] = res['rope_res']
                rope_ss[s] = n
                rope_lift[s] = res['lift']
                peek_hdi_width.append(res['lift_hdi_width'])
                peek_rope_width.append(res['rope_width'])
                if res['rope_res'] != 'U':
                    peek_lift.append(res['lift'])
                
        res = pd.Series(rope_res).value_counts().to_dict()
        peek_avg_hdi_width = np.mean(peek_hdi_width)
        peek_avg_rope_width = np.mean(peek_rope_width)
        peek_avg_lift = np.mean(abs(np.array(peek_lift)))

        sim_res_data_dict = {}
        sim_res_data_dict.update(res)
        sim_res_data_dict['peek_id'] = peek_idx
        sim_res_data_dict['samples'] = n 
        sim_res_data_dict['avg_hdi_width'] = peek_avg_hdi_width
        sim_res_data_dict['avg_rope_width'] = peek_avg_rope_width
        sim_res_data_dict['avg_lift'] = peek_avg_lift
        sim_res_data.append(sim_res_data_dict)
            
        # Summary for each peek
        if pr_peeking:
            helper.fprint("[{}]: Peek #{:,} @ {:,} samples, avg_hdi_width: {:.5f}, avg_rope_width: {:.5f},"
                          " avg_lift: {:.5f}, rope_res: {}"
                         .format(time.strftime("%m-%d %H:%M:%S", time.localtime()), peek_idx, n, 
                                 peek_avg_hdi_width, peek_avg_rope_width, peek_avg_lift, res), fn)

        if 'U' not in rope_res:
            break

    # Simulation summary
    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value: {},"
                  " rope_hdi_ratio: {:.2f}, rope_mde_ratio: {:.3f}, min_rope_half_width: {:.5f},"
                  " n_peeks: {:,}, start: {:,}, step: {:,}, rvs_size: {:,}"
                  .format(muA, muB, sample_size, n_experiments, relative_mde_value, 
                          rope_hdi_ratio, rope_mde_ratio, min_rope_half_width, 
                          n_peeks, start, step, rvs_size), fn)

    winner_ss = []
    winner_lift = []
    if muA == muB:

        for i, r in enumerate(rope_res):
            if r in (['A','B']):
                winner_lift.append(rope_lift[i])

        helper.fprint("[ROPE] {:,} peeks - Type 1 error rate: {:.4f}, error_avg_lift: {:.5f}, rope_res: {}"
                      .format(peek_idx, 1-(res.get('U',0)+res.get('E',0))/n_experiments, 
                              np.mean(abs(np.array(winner_lift))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(rope_res):
            if r == winner:
                winner_ss.append(rope_ss[i])
                winner_lift.append(rope_lift[i])

        helper.fprint("[ROPE] {:,} peeks - Power: {:.4f}, avg_sample_size: {:,.0f},"
                      " sample_size_ratio: {:.2f}%,"
                      " avg_lift: {:.5f}, rope_res: {}"
                      .format(peek_idx, res.get(winner,0)/n_experiments, np.mean(winner_ss), 
                              np.mean(winner_ss)/freq_sample_size*100, np.mean(abs(np.array(winner_lift))), 
                              res), fn)

    # Show fixed sample size result     
    if pr_fixed:
        sim_fixed(muA, muB, sample_size, n_experiments, sim_data, relative_mde_value, rope_hdi_ratio, 
                  rope_mde_ratio, hdi_cred_mass,  rvs_size, fn)

    helper.fprint("[Elasped time]: {}".format(str(timedelta(seconds=time.time()-start_time))), fn)
    helper.fprint("[End time]: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

    return {'sim_res_df': pd.DataFrame(sim_res_data), 
            'rope_res': rope_res,
            'rope_lift': rope_lift}


def sim_fixed(muA, muB, sample_size, n_experiments, sim_data=None, relative_mde_value=0.02, 
              rope_hdi_ratio=0.6, rope_mde_ratio=0.375, hdi_cred_mass=0.95, 
              rvs_size=1000000, fn=None):

    if sim_data is None:
        sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments)

    rope_res = ['U']*n_experiments
    rope_lift = [0]*n_experiments
    for s in range(n_experiments):
        sa = sim_data[s][0][sample_size-1]
        sb = sim_data[s][1][sample_size-1]
        res = rope_core.calc_stat(sa, sample_size, sb, sample_size, relative_mde_value, 
                                  rope_hdi_ratio, rope_mde_ratio, hdi_cred_mass, rvs_size)
        rope_res[s] = res['rope_res']
        rope_lift[s] = res['lift']

    res = pd.Series(rope_res).value_counts().to_dict()

    absolute_mde_value = muA*relative_mde_value
    min_rope_half_width = absolute_mde_value*rope_mde_ratio

    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value: {}, rope_hdi_ratio: {}, rope_mde_ratio: {},"
                  " min_rope_half_width: {:.6f}, rvs_size: {:,}"
                  .format(muA, muB, sample_size, n_experiments, relative_mde_value, 
                          rope_hdi_ratio, rope_mde_ratio, min_rope_half_width, 
                          rvs_size), fn)

    winner_lift = []
    if muA == muB:

        for i, r in enumerate(rope_res):
            if r in (['A','B']):
                winner_lift.append(rope_lift[i])

        helper.fprint("[ROPE] Fixed - Type 1 error rate: {:.4f}, error_avg_lift: {:.5f}, rope_res: {}"
                      .format(1-(res.get('U',0)+res.get('E',0))/n_experiments,
                              np.mean(abs(np.array(winner_lift))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(rope_res):
            if r == winner:
                winner_lift.append(rope_lift[i])

        helper.fprint("[ROPE] Fixed - Power: {:.4f}, avg_lift: {:.5f}, rope_res: {}"
                      .format(res.get(winner,0)/n_experiments, np.mean(abs(np.array(winner_lift))), res), fn)

    return {'rope_res': rope_res, 
            'rope_lift': rope_lift}
