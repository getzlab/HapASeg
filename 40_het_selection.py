import colorama
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import ncls
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

plt.figure(1); plt.clf()
plt.figure(2); plt.clf()
plt.figure(30); plt.clf()
cut20_dens = {}
cut20_lod = {}
cut80_dens = {}
cut80_lod = {}
for depth in [15, 20, 30, 60, 80, 200]:
    # simulate good hets
    cov = s.poisson.rvs(depth, size = 10000)
    A = s.binom.rvs(cov, 0.5)
    B = cov - A

    # simulate bad hets
    bad_cov = s.poisson.rvs(depth, size = 10000)
    bad_frac = np.ones_like(bad_cov).astype(float)
    for i in range(len(bad_frac)):
        bad_frac[i] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    A_bad = s.binom.rvs(bad_cov, bad_frac)
    B_bad = bad_cov - A_bad

    # old criterion: beta density between 0.6 and 0.4
    betafrac = np.diff(s.beta.cdf([0.4, 0.6], A[:, None] + 1, B[:, None] + 1))
    betafrac_bad = np.diff(s.beta.cdf([0.4, 0.6], A_bad[:, None] + 1, B_bad[:, None] + 1))

    # new criterion: log-odds ratio
    betalod = s.beta.logsf(0.5, A + 1, B + 1) - s.beta.logcdf(0.5, A + 1, B + 1)
    betalod_bad = s.beta.logsf(0.5, A_bad + 1, B_bad + 1) - s.beta.logcdf(0.5, A_bad + 1, B_bad + 1)

    # ROC curves
    dens_cdf = np.zeros([1000, 2])
    for i, cut in enumerate(np.linspace(0, 1, 1000)):
        dens_cdf[i, 0] = (betafrac >= cut).mean()
        dens_cdf[i, 1] = (betafrac_bad >= cut).mean()

    lod_cdf = np.zeros([1000, 2])
    for i, cut in enumerate(np.linspace(0, np.abs(np.r_[betalod_bad, betalod]).max(), 1000)):
        lod_cdf[i, 0] = (np.abs(betalod) <= cut).mean()
        lod_cdf[i, 1] = (np.abs(betalod_bad) <= cut).mean()

    plt.figure(30)
    st = plt.step(dens_cdf[:, 1], dens_cdf[:, 0])
    color = st[0].get_color()
    plt.step(lod_cdf[:, 1], lod_cdf[:, 0], color = color, linestyle = ":")

    cut20_dens[depth] = np.linspace(0, 1, 1000)[np.flatnonzero(dens_cdf[:, 1] <= 0.2)[0]]
    cut80_dens[depth] = np.linspace(0, 1, 1000)[np.flatnonzero(dens_cdf[:, 0] <= 0.8)[0]]
    cut20_lod[depth] = np.linspace(0, np.abs(np.r_[betalod_bad, betalod]).max(), 1000)[np.flatnonzero(lod_cdf[:, 1] >= 0.2)[0]]
    cut80_lod_idx = np.flatnonzero(lod_cdf[:, 0] >= 0.8)[0]
    cut80_lod[depth] = np.linspace(0, np.abs(np.r_[betalod_bad, betalod]).max(), 1000)[cut80_lod_idx]

    plt.scatter(lod_cdf[cut80_lod_idx, 1], lod_cdf[cut80_lod_idx, 0], marker = 'x', color = color)
    plt.text(lod_cdf[cut80_lod_idx, 1], lod_cdf[cut80_lod_idx, 0], "{0:.2f}".format(cut80_lod[depth]), color = color)

    plt.figure(1)
    sc = plt.scatter(cov, betafrac, alpha = 0.1, s = 10)
    plt.scatter(depth, np.diff(s.beta.cdf([0.4, 0.6], depth/2 + 1, depth/2 + 1)), color = color, marker = "x")

    cov_range = np.r_[cov.min():cov.max()]
    cov_cum = np.nan*np.ones_like(cov_range)
    for i, c in enumerate(cov_range):
        cov_cum[i] = betafrac[cov >= c].mean()

    plt.figure(2)
    plt.scatter(cov_range, cov_cum)
