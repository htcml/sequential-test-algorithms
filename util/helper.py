import math, pickle
import numpy as np
import scipy.stats as stats


def estimate_toc(mu=0.02, relative_mde_value=0.02, toc_adj_factor=0.00025, pr=False):
    # estimate threshold of caring
    absolute_mde_value = mu*relative_mde_value
    toc = absolute_mde_value*toc_adj_factor

    if pr:
        print("TOC for mu={:.4f}, relative_mde_value={:.4f}, absolute_mde_value={:.4f}:"
              " toc_adj_factor:{:.5f}:"
              .format(mu, relative_mde_value, absolute_mde_value, toc_adj_factor))
        # source: https://help.vwo.com/hc/en-us/articles/360033991153-What-Is-Smart-Decision-
        print("VWO recommended values: high certainty: {:.8f}, balance: {:.8f}, quick learning: {:.8f}"
              .format(absolute_mde_value*0.010, absolute_mde_value*0.075, absolute_mde_value*0.200))
        print("Suggested value for One XP: {:.8f}".format(toc))

    est = {'mu': mu,
           'relative_mde_value': relative_mde_value,
           'absolute_mde_value': absolute_mde_value,
           'toc_adj_factor': toc_adj_factor,
           'toc': toc}

    return est


def sample_size(mu=0.02, relative_mde_value=0.02, alpha=0.05, beta=0.05, variants=2, pr=False):
    # source: equation (4.7), p.77, sample size calculations in clinical research
    # reference: https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/
    # reference: https://vwo.com/blog/ab-test-duration-calculator/
    absolute_mde_value = mu*relative_mde_value

    z_half_alpha = stats.norm.ppf(1-alpha/2)
    z_beta = stats.norm.ppf(1-beta)
    mu2 = mu+absolute_mde_value
    variance = (mu*(1-mu)+mu2*(1-mu2))

    num = math.pow(z_half_alpha+z_beta, 2)*variance
    den = math.pow(absolute_mde_value, 2)

    variant_sample_size = num/den
    total_sample_size = variant_sample_size*variants

    if pr:
        print("Sample size for mu={:.4f}, relative_mde_value={:.4f}, absolute_mde_value={:.4f},"
              " alpha={:.4f}, beta={:.4f}:"
              .format(mu, relative_mde_value, absolute_mde_value, alpha, beta))
        print("variant_sample_size={:,.0f}, total_sample_size={:,.0f}"
              .format(variant_sample_size, total_sample_size))

    ss = {'mu': mu,
          'relative_mde_value': round(relative_mde_value, 5),
          'absolute_mde_value': round(absolute_mde_value, 5),
          'alpha': alpha,
          'beta': beta,
          'variant_sample_size': int(round(variant_sample_size)),
          'total_sample_size': int(round(total_sample_size))}

    return ss


def power(muA, muB, variant_sample_size, alpha=0.05, pr=False):
    # source: equation (4.7), p.77, sample size calculations in clinical research
    # reference: https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/
    # reference: https://vwo.com/blog/ab-test-duration-calculator/
    absolute_mde_value = muB-muA
    relative_mde_value = absolute_mde_value/muA

    z_half_alpha = stats.norm.ppf(1-alpha/2)
    variance = (muA*(1-muA)+muB*(1-muB))

    z_beta = z_half_alpha - math.pow(variant_sample_size*math.pow(absolute_mde_value, 2)/variance, 0.5)
    beta = stats.norm.cdf(z_beta)

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


def sim_binomial_seq(mu_lst, sample_size, n_experiments, cumsum=True, random_seed=None): 

    if random_seed:
        np.random.seed(random_seed)

    sim_data = [] 
    for i in range(n_experiments):
        experiment = []
        for mu in mu_lst: 
            if cumsum:
                experiment.append(np.cumsum(np.random.binomial(1, mu, size=sample_size)))
            else:
                experiment.append(np.random.binomial(1, mu, size=sample_size))
        sim_data.append(experiment)

    return sim_data


def sample_variance(samples):

    sample_mean = np.mean(samples)
    tot = 0 
    for s in samples:
        tot += (s-sample_mean)**2

    return tot/(len(samples)-1)


def fprint(str, fn):
    if fn:
        f = open(fn, 'a+')
        f.write(str+'\n')
        f.close()
    else:
        print(str)


def pickle_dump(obj, fn):
    file = open(fn, 'wb')
    pickle.dump(obj, file)
    file.close()


def pickle_load(fn):
    file = open(fn, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj
