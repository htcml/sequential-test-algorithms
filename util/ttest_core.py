import numpy as np
import scipy.stats as stats

def calc_stat(sample_a, sample_b, alpha=0.05):

    n_a = len(sample_a)
    n_b = len(sample_b)
    s_a = np.sum(sample_a)
    s_b = np.sum(sample_b)
    cvr_a = np.mean(sample_a)
    cvr_b = np.mean(sample_b)
    effect = cvr_b - cvr_a
    var_a = np.var(sample_a, ddof=1) # unbiased sample variance
    var_b = np.var(sample_b, ddof=1) # unbiased sample variance
    sigma = np.sqrt(var_a/n_a+var_b/n_b)

    #p_value = stats.ttest_ind(a=sample_a, b=sample_b, equal_var=False)[1]
    t_statistic = effect/sigma
    df0 = (var_a/n_a+var_b/n_b)**2
    df1 = (var_a**2/(n_a**2*(n_a-1)))+(var_b**2/(n_b**2*(n_b-1)))
    df = df0/df1
    p_value = stats.t.cdf(-abs(t_statistic), df)*2

    decision_boundary = stats.t.ppf(1-alpha/2, df)*sigma
    effect_ci_lb = effect - decision_boundary
    effect_ci_ub = effect + decision_boundary

    if p_value < alpha:
        if effect > 0:
            ttest_res = 'B'
        else:
            ttest_res = 'A'
    else:
        ttest_res = 'U'

    return {'cvr_data': [s_a, n_a, s_b, n_b],
            'cvr_a': cvr_a,
            'cvr_b': cvr_b,
            'effect': effect,
            'var_a': var_a,
            'var_b': var_b,
            'sigma': sigma,
            'df': df,
            't_statistic': t_statistic,
            'p_value': p_value, 
            'decision_boundary': decision_boundary,
            'effect_ci': [effect_ci_lb, effect_ci_ub], 
            'ttest_res': ttest_res}

def estimate_sample_size(mu=0.02, relative_mde_value=0.02, alpha=0.05, beta=0.05, variants=2, pr=True):
    # source: equation (4.7), p.77, sample size calculations in clinical research
    # reference: https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/
    # reference: https://vwo.com/blog/ab-test-duration-calculator/
    muA = mu
    muB = muA*(1+relative_mde_value)
    effect = muB - muA
    variance = muA*(1-muA)+muB*(1-muB)
    # z_k is the upper k-th percentile of the standard normal distribution
    z_half_alpha = stats.norm.ppf(1-alpha/2)
    z_beta = stats.norm.ppf(1-beta)

    variant_sample_size = ((z_half_alpha+z_beta)**2)*variance/(effect**2)
    total_sample_size = variant_sample_size*variants

    if pr:
        print("muA: {:.5f}, muB: {:.5f}, relative_mde_value: {:.5f}, variant_sample_size: {:,}"
              .format(muA, muB, relative_mde_value, int(variant_sample_size)))

    return {'mu': mu,
            'relative_mde_value': round(relative_mde_value, 5),
            'absolute_mde_value': round(effect, 5),
            'alpha': alpha,
            'beta': beta,
            'variant_sample_size': int(variant_sample_size),
            'total_sample_size': int(total_sample_size)}

def estimate_power(muA, muB, variant_sample_size, alpha=0.05, pr=False):
    # source: equation (4.7), p.77, sample size calculations in clinical research
    # reference: https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/
    # reference: https://vwo.com/blog/ab-test-duration-calculator/
    absolute_mde_value = muB-muA
    relative_mde_value = absolute_mde_value/muA

    z_half_alpha = stats.norm.ppf(1-alpha/2)
    variance = muA*(1-muA)+muB*(1-muB)

    z_beta = (variant_sample_size*(absolute_mde_value**2)/variance)**0.5 - z_half_alpha
    beta = 1-stats.norm.cdf(z_beta)

    if pr:
        print("Power for muA={:.4f}, muB={:.4f}, relative_mde_value={:.4f},"
              " variant_sample_size={:,.0f}, alpha={:.4f}:"
              .format(muA, muB, relative_mde_value, variant_sample_size, alpha))
        print("{:.4f}".format(1-beta))

    ss = {'muA': muA,
          'muB': muB,
          'relative_mde_value': round(relative_mde_value, 5),
          'absolute_mde_value': round(absolute_mde_value, 5),
          'variant_sample_size': int(round(variant_sample_size)),
          'alpha': round(alpha, 5),
          'beta': round(beta, 5),
          'power': round(1-beta, 5)}

    return ss

