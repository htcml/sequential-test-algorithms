import numpy as np
import scipy.stats as stats

def calc_stat(sa, na, sb, nb, alpha=0.05, prev_p_value=1, prev_effect_ci=[-1000,1000]):

    cvr_a = sa/na
    cvr_b = sb/nb
    effect = cvr_b - cvr_a
    V = cvr_a*(1-cvr_a)/na + cvr_b*(1-cvr_b)/nb
    tau = 0.0001       

    try:
        Lambda = np.sqrt(V/(V+tau))*np.exp((tau/(2*V*(V+tau)))*(effect**2))
    except:
        Lambda = 1

    try:
        decision_boundary = np.sqrt((V*(V+tau)/tau)*(-2*np.log(alpha)-np.log(V/(V+tau))))
    except:
        decision_boundary = 1000

    # compute always valid p-values
    p_value = min(prev_p_value, 1/Lambda)

    # generate test result using p_value
    if p_value < alpha:          
       if effect > 0:
            msprt_res = 'B'
       else:
            msprt_res = 'A'
    else:
        msprt_res = 'U'

    # compute always valid confidence interval
    curr_effect_ci_lb = effect - decision_boundary
    curr_effect_ci_ub = effect + decision_boundary

    effect_ci_lb = max(prev_effect_ci[0], curr_effect_ci_lb)
    effect_ci_ub = min(prev_effect_ci[1], curr_effect_ci_ub)

    return {'cvr_data': [sa, na, sb, nb],
            'cvr_a': cvr_a,
            'cvr_b': cvr_b,
            'effect': effect,
            'tau': tau,
            'V': V,
            'Lambda': Lambda,
            'decision_boundary': decision_boundary,
            'p_value': p_value, # Optimizely's always valid p-value
            'effect_ci': [effect_ci_lb, effect_ci_ub], # Optimizely's always valid confidence-interval
            'msprt_res': msprt_res}


def calc_ab(data, alpha=0.05):
    
    sa = np.sum(data[0])
    na = len(data[0])

    sb = np.sum(data[1])
    nb = len(data[1])

    return calc_stat(sa, na, sb, nb, alpha)


def estimate_sample_size(baseline_cvr, relative_mde_value, alpha=0.05, variants=2, pr=True):

    muA = baseline_cvr 
    muB = muA*(1+relative_mde_value)
    var_a = muA*(1-muA)
    var_b = muB*(1-muB)
    effect = muB - muA

    tau = 0.0001
    A = (var_a+var_b)/effect**2
    B = -2*tau*np.exp(effect**2/tau)*np.log(alpha)
    C = (effect**2)*(alpha**2)
    variant_sample_size = int(A*(np.log(B/C)-1))
    #variant_sample_size = int(((var_a+var_b)/effect**2)*(np.log(-2*np.log(alpha))-2*np.log(alpha)))
    total_sample_size = variant_sample_size*variants

    if pr:
        print("muA: {:.5f}, muB: {:.5f}, relative_mde_value: {:.5f}, variant_sample_size: {:,}"
              .format(muA, muB, relative_mde_value, variant_sample_size))

    return {'baseline_cvr': baseline_cvr,
            'relative_mde_value': round(relative_mde_value, 5),
            'absolute_mde_value': round(effect, 5),
            'alpha': alpha,
            'variant_sample_size': int(variant_sample_size),
            'total_sample_size': int(total_sample_size)}


def estimate_sample_size_threshold(baseline_cvr, relative_mde_value, alpha=0.05, beta=0.05, pr=True):

    ss = estimate_sample_size(baseline_cvr, relative_mde_value, alpha, pr=False)
    threshold = int((0.35-0.79*np.log(beta))*ss['variant_sample_size'])

    if pr:
        print("baseline_cvr: {:.5f}, relative_mde_value: {:.5f}, threshold: {:,}"
              .format(baseline_cvr, relative_mde_value, threshold))

    return {'baseline_cvr': baseline_cvr, 
            'relative_mde_value': round(relative_mde_value, 5),
            'absolute_mde_value': round(baseline_cvr*relative_mde_value, 5),
            'alpha': alpha,
            'beta': beta,
            'threshold': int(threshold)}


def estimate_power(baseline_cvr, relative_mde_value, current_sample_size, alpha=0.05):

    ss = estimate_sample_size(baseline_cvr, relative_mde_value, alpha, pr=False)['variant_sample_size']
    return np.exp((0.35-current_sample_size/ss)/0.79)
