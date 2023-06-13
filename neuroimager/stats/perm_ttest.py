import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# define a permutation_test function
# p value here is from two-tailed analysis
def perm_ttest_ind(g1, g2, n_reps=5000, plot=True):
    """
    params:
            g1: array of group1
            g2: array of group2
            n_reps: the number of permutation
            plot: whether to plot the permutation distribution
    returns:
            perm_p: two-tailed p value of permutation test
            obs_diff: observed mean difference
            perm_diff: np.array. Contains permuted difference.
    """
    obs_diff = np.mean(g1) - np.mean(g2)
    data = np.append(g1, g2)
    g1_len = len(g1)
    perm_diff = np.zeros(n_reps)
    for i in range(n_reps):
        perm_data = np.random.permutation(data)
        perm_g1 = perm_data[:g1_len]
        perm_g2 = perm_data[g1_len:]
        perm_diff[i] = perm_g1.mean() - perm_g2.mean()
    # two-tailed p
    perm_p = np.count_nonzero(abs(perm_diff) > abs(obs_diff)) / n_reps
    if plot:
        perm_plot(perm_p, obs_diff, perm_diff, arrow_shift=(0.05, 5))
    return perm_p, obs_diff, perm_diff


# plot the result of permutation test
def perm_plot(perm_p, observed, perm_array, arrow_shift=(0.05, 5)):
    """
    params:
        output from function permutation_test()
        arrow_shift: the start position of the arrow relative to the observed value
    """
    sns.distplot(perm_array)
    plt.plot(observed, 0, "ro")
    plt.annotate(
        "p={}".format(perm_p),
        ha="center",
        va="bottom",
        xytext=(observed + arrow_shift[0], arrow_shift[1]),
        xy=(observed, 0),
        arrowprops={"facecolor": "red", "shrink": 0.05},
    )
    plt.show()
