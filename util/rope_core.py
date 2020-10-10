import numpy as np
import scipy.stats as stats


def hdi(sample_vec, hdi_cred_mass=0.95):
    # compute highest posterior density interval
    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    hdi_len = int(np.floor(hdi_cred_mass*len(sorted_pts)))
    hdi_width = sorted_pts[hdi_len:] - sorted_pts[:(len(sorted_pts) - hdi_len)]

    start_idx = np.argmin(hdi_width) # find the shortest 95% HDI
    hdi_min = sorted_pts[start_idx]
    hdi_max = sorted_pts[start_idx+hdi_len]

    return hdi_min, hdi_max


def calc_stat(sa, na, sb, nb, relative_mde_value=0.02, 
              rope_hdi_ratio=0.6, rope_mde_ratio=0.375, 
              hdi_cred_mass=0.95, rvs_size=1000000):

    alpha_a = sa + 1
    beta_a = 1 + na - sa
    alpha_b = sb + 1
    beta_b = 1 + nb - sb

    rvs_a = stats.beta.rvs(alpha_a, beta_a, size=rvs_size)
    rvs_b = stats.beta.rvs(alpha_b, beta_b, size=rvs_size)

    prob_b_gt_a = np.mean(rvs_b>rvs_a)
    cvr_a = np.mean(rvs_a)
    cvr_b = np.mean(rvs_b)
    lift = cvr_b - cvr_a
    lift_rvs = rvs_b - rvs_a

    lift_hdi_min, lift_hdi_max = hdi(lift_rvs, hdi_cred_mass)
    rope_half_width = (lift_hdi_max-lift_hdi_min)*rope_hdi_ratio/2

    min_rope_half_width = cvr_a*relative_mde_value*rope_mde_ratio

    if rope_half_width < min_rope_half_width:
        rope_half_width = min_rope_half_width

    rope_min, rope_max = -1*rope_half_width, rope_half_width

    # [rope]....[hdi]
    if rope_max < lift_hdi_min:
        rope_res = 'B'
    # [hdi]....[rope]
    elif rope_min > lift_hdi_max:
        rope_res = 'A'
    # [rope_min..[hdi]..rope_max]
    elif rope_min < lift_hdi_min and lift_hdi_max < rope_max:
        rope_res = 'E'
    else:
        rope_res = 'U'

    return {'cvr_data': [sa, na, sb, nb],
            'beta_params': [alpha_a, beta_a, alpha_b, beta_b],
            'cvr_a': cvr_a,
            'cvr_b': cvr_b,
            'prob_b_gt_a': prob_b_gt_a,
            'lift': lift,
            'rope_hdi_ratio': rope_hdi_ratio,
            'rope_mde_ratio': rope_mde_ratio,
            'min_rope_half_width': min_rope_half_width,
            'rope': (rope_min, rope_max),
            'rope_width': rope_max-rope_min,
            'lift_hdi': (lift_hdi_min, lift_hdi_max),
            'lift_hdi_width': lift_hdi_max-lift_hdi_min,
            'rope_res': rope_res}


def calc_ab(data, relative_mde_value=0.02,
            rope_hdi_ratio=0.6, rope_mde_ratio=0.375,
            hdi_cred_mass=0.95, rvs_size=1000000):
    
    sa = np.sum(data[0])
    na = len(data[0])

    sb = np.sum(data[1])
    nb = len(data[1])

    return calc_stat(sa, na, sb, nb, relative_mde_value, 
                     rope_hdi_ratio, rope_mde_ratio, hdi_cred_mass, rvs_size)

