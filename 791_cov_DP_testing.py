#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy_groupies as npg
import pandas as pd
import scipy.stats as s
import scipy.special as ss
import sortedcontainers as sc
import os
import tqdm
import matplotlib


# In[2]:


os.environ["CAPY_REF_FA"] = "/home/opriebe/data/ref/hg19/Homo_sapiens_assembly19.fasta"
import hapaseg.coverage_MCMC as mcmc_cov
import hapaseg.NB_coverage_MCMC as nb_cov
from capy import mut, seq


# In[3]:


import hapaseg.coverage_DP as dp_cov


# In[4]:


#loading cov_seg_df
cov_seg_df = pd.read_pickle('./cov_MCMC_df')


# In[5]:


beta = np.load('./beta_save.npy')


# In[6]:


cov_dp = dp_cov.Cov_DP(cov_seg_df, beta)


# In[7]:


cov_dp.run()


# In[ ]:




