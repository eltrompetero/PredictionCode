# ====================================================================================== #
# Calculation of info theoretic quantities on C. elegans data set.
# 
# This can be run with Python 3.
#
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
from entropy.entropy import joint_p_mat, MI
from multiprocess import Pool



@np.errstate(divide='ignore', invalid='ignore')
def entropy_with_bins(x, bins):
    """Calculate entropy as a function of the number of bins used to equally divide
    the interval.
    
    Parameters
    ----------
    x : ndarray
    bins list
        List of no. of bins to try.
        
    Returns
    -------
    ndarray
        Entropy in bits for each bin number tried.
    """
    
    S = np.zeros(len(bins))
    for counter, n in enumerate(bins):
        digx = np.digitize(x, np.linspace(x.min(), x.max(), n+1)[:-1])
        p = np.bincount(digx)
        p = p/p.sum()
        S[counter] = -np.nansum(p * np.log2(p))
    return S

def variance_cost(logwidth, y):
    """Variance of bin counts for three bins, one corresponding to negative values
    that are down states, near zero values that are flat states, and positive values
    that are up states.
    
    Parameters
    ----------
    logwidth : ndarray
        Log of widths to try for binning.
    y : ndarray
        Values to bin.

    Returns
    -------
    ndarray
        Variance of the histogram.
    """

    if np.nanmin(y) >= -np.exp(logwidth): return 1e30
    
    # remember that nothing is in 0 bin from the way that digitize is defined
    counts = np.bincount(np.digitize(y[~np.isnan(y)],
                                     [np.nanmin(y),
                                      -np.exp(logwidth),
                                      np.exp(logwidth)]))[1:]
        
    if counts.size!=3: return 1e30
    return np.var(counts)

def behavior_mi(der, vel, cur, bins_vel=4, bins_cur=4):
    """Mutual information between Hallinen et al. behavioral data on velocity and
    curvature with neural calcium derivative.

    Parameters
    ----------
    der : ndarray
    vel : ndarray
        (n_neurons, n_time)
    cur : ndarray
        (n_neurons, n_time)
    bins_vel : int, 4
        Number of bins to use for discretizing worm velocity. This is considered
        elsewhere in the notebook.
    bins_cur : int, 4

    Returns
    -------
    list
    list
    list
    float
    float
    """

    # find optimal threshold point for flat/down/up states
    logwidth = np.linspace(-7, -1, 100)
    c = np.array([variance_cost(w, der)for w in logwidth])

    # discretize derivative such that it indicates flat/down/up
    der = np.digitize(der,
                      [der.min(),
                       -np.exp(logwidth[c.argmin()]),
                       np.exp(logwidth[c.argmin()])]).reshape(der.shape)

    # digitize velocity
    vel = np.digitize(vel, np.linspace(vel.min(), vel.max(), bins_vel)[:-1])

    # digitize curvature
    cur = np.digitize(cur, np.linspace(cur.min(), cur.max(), bins_cur)[:-1])

    # mutual info of individual neuron
    indmi = [np.zeros(der.shape[0]), np.zeros(der.shape[0])]
    for i, d in enumerate(der):
        x = d
        y = vel.copy()
        indmi[0][i] = MI(joint_p_mat(np.vstack((x,y)).T, [0], [1]))

        y = cur.copy()
        indmi[1][i] = MI(joint_p_mat(np.vstack((x,y)).T, [0], [1]))

    # fine and coarse-grained MI for velocity
    x = np.vstack([np.bincount(c, minlength=4)[1:] for c in der.T.astype(int)])
    x = np.sort(x, axis=1)
    y = vel.copy()
    syncmi = [MI(joint_p_mat(np.hstack((x,y[:,None])), [0,1,2], [3]))]
    csyncmi = [MI(joint_p_mat(np.vstack((x[:,-1],y)).T, [0], [1]))]

    # fine and coarse-grained MI for curvature
    y = cur.copy()
    syncmi.append(MI(joint_p_mat(np.hstack((x,y[:,None])), [0,1,2], [3])))
    csyncmi.append(MI(joint_p_mat(np.vstack((x[:,-1],y)).T, [0], [1])))

    p = np.unique(vel, return_counts=True)[1]
    p = p / p.sum()
    vel_S = -p.dot(np.log2(p))

    p = np.unique(cur, return_counts=True)[1]
    p = p / p.sum()
    cur_S = -p.dot(np.log2(p))
    
    return indmi, syncmi, csyncmi, vel_S, cur_S
