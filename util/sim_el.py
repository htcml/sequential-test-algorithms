import time
from datetime import timedelta
import numpy as np
import pandas as pd
import util.helper as helper
import util.el_core as el_core


def sim_peeking(muA, muB, sample_size, n_experiments, n_peeks=-1, start=None, 
                relative_mde_value=0.02, toc_adj_factor=0.00025,
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

    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value: {}, toc_adj_factor: {:.5f}, n_peeks: {:,}, start: {:,}, step: {:,}"
                  .format(muA, muB, sample_size, n_experiments, relative_mde_value, toc_adj_factor, 
                          n_peeks, start, step), fn)

    sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments)

    sim_res_data = []
    el_res = ['U']*n_experiments
    el_ss = [0]*n_experiments
    el_lift = [0]*n_experiments
    peek_idx = 0
    for n in range(start, sample_size+1, step):
        peek_idx += 1
        peek_lift = []
        for s in range(n_experiments):
            if el_res[s] == 'U':
                sa = sim_data[s][0][n-1]
                sb = sim_data[s][1][n-1]
                res = el_core.calc_stat(sa, n, sb, n, relative_mde_value, toc_adj_factor) 
                el_res[s] = res['el_res']
                el_ss[s] = n
                el_lift[s] = res['lift']
                if res['el_res'] != 'U':
                    peek_lift.append(res['lift'])

        res = pd.Series(el_res).value_counts().to_dict()
        peek_avg_lift = np.mean(abs(np.array(peek_lift)))

        sim_res_data_dict = {}
        sim_res_data_dict.update(res)
        sim_res_data_dict['peek_id'] = peek_idx
        sim_res_data_dict['samples'] = n 
        sim_res_data_dict['avg_lift'] = peek_avg_lift
        sim_res_data.append(sim_res_data_dict)
            
        helper.fprint("[{}]: Peek #{:,} @ {:,} samples, avg_lift: {:.5f}, el_res: {}"
              .format(time.strftime("%m-%d %H:%M:%S", time.localtime()), peek_idx, n,
                                    peek_avg_lift, res), fn)

        if ('U' not in el_res):
            break
        
    # Summary
    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value: {}, toc_adj_factor: {:.5f}, n_peeks: {:,}, start: {:,}, step: {:,}"
                  .format(muA, muB, sample_size, n_experiments, 
                          relative_mde_value, toc_adj_factor, n_peeks, start, step), fn)

    winner_ss = []
    winner_lift = []
    if muA == muB:

        for i, r in enumerate(el_res):
            if r in (['A','B']):
                winner_lift.append(el_lift[i])

        if pr_peeking:
            helper.fprint("[EL] {:,} peeks - Type 1 error rate: {:.4f}, error_avg_lift: {:.5f}, el_res: {}"
                          .format(peek_idx, 1-(res.get('U',0)+res.get('E',0))/n_experiments,
                                  np.mean(abs(np.array(winner_lift))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(el_res):
            if r == winner:
                winner_ss.append(el_ss[i])
                winner_lift.append(el_lift[i])

        if pr_peeking:
            helper.fprint("[EL] {:,} peeks - Power: {:.4f}, avg_sample_size: {:,.0f},"
                          "sample_size_ratio: {:.2f}%, avg_lift: {:.5f}, el_res: {}"
                          .format(peek_idx, res.get(winner,0)/n_experiments, np.mean(winner_ss),
                                  np.mean(winner_ss)/freq_sample_size*100, 
                                  np.mean(abs(np.array(winner_lift))), res), fn)

    if pr_fixed:
        sim_fixed(muA, muB, sample_size, n_experiments, sim_data, relative_mde_value)

    helper.fprint("[Elasped time]: {}".format(str(timedelta(seconds=time.time()-start_time))), fn)
    helper.fprint("[End time]: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

    return {'sim_res_df': pd.DataFrame(sim_res_data),
            'el_res': el_res,
            'el_lift': el_lift}


def sim_fixed(muA, muB, sample_size, n_experiments, sim_data=None, 
              relative_mde_value=0.02, toc_adj_factor=0.00025, fn=None):

    if sim_data is None:
        sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments)

    el_res = ['U']*n_experiments
    el_lift = [0]*n_experiments
    for s in range(n_experiments):
        sa = sim_data[s][0][sample_size-1]
        sb = sim_data[s][1][sample_size-1]
        res = el_core.calc_stat(sa, sample_size, sb, sample_size, relative_mde_value, toc_adj_factor)
        el_res[s] = res['el_res']
        el_lift[s] = res['lift']

    res = pd.Series(el_res).value_counts().to_dict()

    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value: {}, toc: {:.10f}"
                  .format(muA, muB, sample_size, n_experiments, relative_mde_value, toc), fn)

    winner_lift = []
    if muA == muB:

        for i, r in enumerate(el_res):
            if r in (['A','B']):
                winner_lift.append(el_lift[i])

        helper.fprint("[EL] Fixed - Type 1 error rate: {:.4f}, error_avg_lift: {:.5f}, el_res: {}"
                      .format(1-(res.get('U',0)+res.get('E',0))/n_experiments,
                              np.mean(abs(np.array(winner_lift))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(el_res):
            if r == winner:
                winner_lift.append(el_lift[i])

        helper.fprint("[EL] Fixed - Power: {:.4f}, avg_lift: {:.5f}, el_res: {}"
                      .format(res.get(winner,0)/n_experiments, 
                              np.mean(abs(np.array(winner_lift))), res), fn)

    return {'el_res': el_res,
            'el_lift': el_lift}
