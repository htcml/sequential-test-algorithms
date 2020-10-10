import time
from datetime import timedelta
import numpy as np
import pandas as pd
import util.helper as helper
import util.ttest_core as ttest_core


def sim_peeking(muA, muB, sample_size, n_experiments, n_peeks=-1, start=None, 
                alpha=0.05, relative_mde_value=0.02, 
                burnIn=300, random_seed=-1,
                pr_peek=1, pr_fixed=False, fn=None):

    if random_seed > 0:
        np.random.seed(random_seed)

    start_time = time.time()
    helper.fprint("[Start time]: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

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

    # print parameter settings
    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value:{:.5f}, alpha: {:.3f}, burnIn:{:,}, random_seed:{},"
                  " n_peeks: {:,}, start: {:,}, step: {:,}"
                  .format(muA, muB, sample_size, n_experiments, 
                          relative_mde_value, alpha, burnIn, random_seed,
                          n_peeks, start, step), fn)

    sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments, 
                                       cumsum=False)

    sim_res_data = []
    ttest_res = ['U']*n_experiments
    ttest_ss = [0]*n_experiments
    ttest_effect = [0]*n_experiments
    peek_idx = 0
    for n in range(start, sample_size+1, step):  
        peek_idx += 1
        peek_effect = []
        for s in range(n_experiments):
            if ttest_res[s] == 'U':
                sample_a = sim_data[s][0][:n]
                sample_b = sim_data[s][1][:n]
                ret = ttest_core.calc_stat(sample_a, sample_b, alpha, burnIn)
                cvr_a = ret['cvr_a']    
                cvr_b = ret['cvr_b']
                effect = ret['effect']
                ttest_res[s] = ret['ttest_res']
                ttest_ss[s] = n
                ttest_effect[s] = effect
                peek_effect.append(effect)
            
                if (ret['p_value'] < alpha):
                    if cvr_a > cvr_b:
                        ttest_res[s] = 'A'
                    else:
                        ttest_res[s] = 'B'
                    peek_effect.append(effect)

        res = pd.Series(ttest_res).value_counts().to_dict()
        peek_avg_effect = np.mean(abs(np.array(peek_effect)))

        sim_res_data_dict = {}
        sim_res_data_dict.update(res)
        sim_res_data_dict['peek_id'] = peek_idx
        sim_res_data_dict['samples'] = n
        sim_res_data_dict['avg_effect'] = peek_avg_effect
        sim_res_data.append(sim_res_data_dict)

        # print peek summary
        if pr_peek > 0 and peek_idx%pr_peek == 0:
            helper.fprint("[{}]: Peek #{} @ {:,} samples, avg_effect: {:.5f}, ttest_res: {}"
                          .format(time.strftime("%m-%d %H:%M:%S", time.localtime()), 
                                  peek_idx, n, peek_avg_effect, res), fn)
                
        if 'U' not in ttest_res:
            break

    # print parameter settings
    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,},"
                  " relative_mde_value:{:.5f}, alpha: {:.3f}, burnIn:{:,}, random_seed:{},"
                  " n_peeks: {:,}, start: {:,}, step: {:,}"
                  .format(muA, muB, sample_size, n_experiments, 
                          relative_mde_value, alpha, burnIn, random_seed,
                          n_peeks, start, step), fn)


    # print simulation summary
    winner_ss = []
    winner_effect = []
    if muA == muB:

        for i, r in enumerate(ttest_res):
            if r in (['A','B']):
                winner_effect.append(ttest_effect[i])

        helper.fprint("[T-Test] {:,} peeks - Type 1 error rate: {:.4f},"
                      "error_avg_effect: {:.5f}, ttest_res: {}"
                      .format(peek_idx, 1-(res.get('U',0)+res.get('E',0))/n_experiments,
                              np.mean(abs(np.array(winner_effect))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(ttest_res):
            if r == winner:
                winner_ss.append(ttest_ss[i])
                winner_effect.append(ttest_effect[i])

        helper.fprint("[T-Test] {:,} peeks - Power: {:.4f}, avg_sample_size: {:,.0f}, sample_size_ratio: {:.2f}%,"
                      " avg_effect: {:.5f}, ttest_res: {}"
                      .format(peek_idx, res.get(winner,0)/n_experiments, np.mean(winner_ss),
                              np.mean(winner_ss)/freq_sample_size*100, 
                              np.mean(abs(np.array(winner_effect))), res), fn)

    if pr_fixed:
        sim_fixed(muA, muB, sample_size, n_experiments, sim_data, alpha) 

    helper.fprint("[Elasped time]: {}".format(str(timedelta(seconds=time.time()-start_time))), fn)
    helper.fprint("[End time]: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), fn)

    return {'sim_res_df': pd.DataFrame(sim_res_data),
            'ttest_res': ttest_res,
            'ttest_effect': ttest_effect}


def sim_fixed(muA, muB, sample_size, n_experiments, sim_data=None, alpha=0.05, fn=None):

    if sim_data is None:
        sim_data = helper.sim_binomial_seq([muA, muB], sample_size=sample_size, n_experiments=n_experiments, 
                                           cumsum=False)

    ttest_res = ['U']*n_experiments
    ttest_effect = [0]*n_experiments
    for s in range(n_experiments):
        sample_a = sim_data[s][0]
        sample_b = sim_data[s][1]
        ret = ttest_core.calc_stat(sample_a, sample_b, alpha)
        ttest_effect[s] = ret['effect']

        if (ret['p_value'] < alpha):
            if ret['cvr_a'] > ret['cvr_b']:
                ttest_res[s] = 'A'
            else:
                ttest_res[s] = 'B'

    helper.fprint("[Parameters]: muA: {}, muB: {}, sample_size: {:,}, n_experiments: {:,}"
                  .format(muA, muB, sample_size, n_experiments), fn)

    winner_effect = []
    res = pd.Series(ttest_res).value_counts().to_dict()

    if muA == muB:

        for i, r in enumerate(ttest_res):
            if r in (['A','B']):
                winner_effect.append(ttest_effect[i])

        helper.fprint("[T-Test] Fixed - Type 1 error rate: {:.4f}, error_avg_effect: {:.5f}, ttest_res: {}"
                      .format(1-(res.get('U',0)+res.get('E',0))/n_experiments,
                              np.mean(abs(np.array(winner_effect))), res), fn)
    else:
        if muA > muB:
            winner = 'A'
        else:
            winner = 'B'

        for i, r in enumerate(ttest_res):
            if r == winner:
                winner_effect.append(ttest_effect[i])

        helper.fprint("[T-Test] Fixed - Power: {:.4f}, avg_effect: {:.5f}, ttest_res: {}"
                      .format(res.get(winner,0)/n_experiments, np.mean(abs(np.array(winner_effect))), res), fn)

    return {'ttest_res': ttest_res,
            'ttest_effect': ttest_effect}

