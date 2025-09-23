import numpy as np
from .tables import get_panav_distns
from sklearn.linear_model import BayesianRidge

REFERENCE_TABLE = get_panav_distns()

def gaussian(x, mu, sigma):
    if sigma == 0:
        return 0.0  # or handle as special case
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return prefactor * np.exp(exponent)

def get_ss_probs(row):
    scores=[]
    for ss in ['C', 'H', 'E']:
        mean, std = REFERENCE_TABLE[row['Comp_ID']][ss][row['Atom_ID']]
        scores.append(gaussian(row['Val'], mean, std))
        
    probs = scores / np.sum(scores)
    selected = ['C','H','E'][np.argmax(scores)]
    return probs, selected

def get_offset_panav(row, corresponding_ha):
    if len(corresponding_ha)==0:
        return np.nan
    else:
        ss = corresponding_ha.iloc[0]['ss_max']
        return row['Val'] - REFERENCE_TABLE[row['Comp_ID']][ss][row['Atom_ID']][0]

def apply_offset(row,offsets):
    if row['Atom_ID'] in offsets.keys():
        return row['Val'] - offsets[row['Atom_ID']]
    else:
        return row['Val']

def chop_at_jumps(data, jump=5):
    data = data[~np.isnan(data)]
    data = np.sort(data)

    n = len(data)
    mid = n // 2
    if n % 2 == 1:
        median_idx = mid
    else:
        median_idx = mid - 1

    lo = hi = median_idx

    while hi + 1 < n and data[hi + 1] - data[hi] <= jump:
        hi += 1

    while lo - 1 >= 0 and data[lo] - data[lo - 1] <= jump:
        lo -= 1

    return data[lo], data[hi]


def robust_fit(x, y, 
        prior_intercept=0.757, prior_slope=-0.386):
    """
    Bayesian linear regression with priors on intercept and slope.
    
    Parameters:
    x, y: input data arrays
    prior_intercept: expected intercept value
    prior_slope: expected slope value  
    alpha_init: precision of prior on intercept (higher = stronger prior)
    lambda_init: precision of prior on slope (higher = stronger prior)
    
    Returns:
    intercept, slope: fitted parameters
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    x = x.reshape(-1, 1)
    
    # Bayesian linear regression with priors
    bayes_reg = BayesianRidge(
        compute_score=True,
        fit_intercept=True
    )
    
    bayes_reg.fit(x, y)
    score = bayes_reg.score(x, y)
    
    if score > 0:  # Any correlation
        return bayes_reg.intercept_
    else:
        return 0
