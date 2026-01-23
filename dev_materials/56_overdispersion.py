import colorama
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import ncls
import numpy as np
import pandas as pd
import scipy.stats as s
import scipy.sparse as sp
import scipy.special as ss
import sortedcontainers as sc

# step 1: see if there's any overdispersion in the normal
N = pd.read_csv("exome/6_C1D1_META.normal.tsv", sep = "\t")

# 1.1: look at overall fit
a, b, _, _ = s.beta.fit(N["REF_COUNT"]/(N["ALT_COUNT"] + N["REF_COUNT"]), floc = 0, fscale = 1)

(a + b)/(N["REF_COUNT"] + N["ALT_COUNT"]).mean()

# 1.2: bin every 10/30/100 SNPs
B = {}

for bin_count in [10, 30, 100, 300, 1000]:
    N["group"] = N.index//bin_count
    B[bin_count] = pd.DataFrame({ 
      "group" : np.unique(N.index//bin_count),
      "a" : np.nan,
      "b" : np.nan,
      "mean_a" : np.nan,
      "mean_b" : np.nan
    })
    B_df = B[bin_count]
    for idx, g in N.groupby("group"):
        B_df.loc[idx, "a"], B_df.loc[idx, "b"], _, _ = s.beta.fit(g["REF_COUNT"]/(g["REF_COUNT"] + g["ALT_COUNT"]), floc = 0, fscale = 1)
        B_df.loc[idx, "mean_a"] = g["REF_COUNT"].mean()
        B_df.loc[idx, "mean_b"] = g["ALT_COUNT"].mean()

plt.figure(1); plt.clf()
_, axs = plt.subplots(5, 1, num = 1)
for i, Bc in enumerate(B.values()):
    axs[i].scatter(Bc["mean_a"] + Bc["mean_b"], Bc["a"] + Bc["b"], alpha = 0.1)
    axs[i].set_xlim([100, 400])
    axs[i].set_ylim([0, 400])
    axs[i].plot(axs[i].get_xlim(), axs[i].get_xlim(), color = 'k', linestyle = ":")

axs[-1].set_xlabel("Mean coverage")
axs[1].set_ylabel("Estimated coverage from beta dist.")

plt.figure(2); plt.clf()
for Bc in B.values():
    print(np.mean((Bc["a"] + Bc["b"])/(Bc["mean_a"] + Bc["mean_b"])))
    plt.hist((Bc["a"] + Bc["b"])/(Bc["mean_a"] + Bc["mean_b"]), np.linspace(0, 3, 20))
