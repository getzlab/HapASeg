import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt

beta_A, beta_B = np.meshgrid(np.r_[1:10000:50], np.r_[1:10000:50])

m1 = beta_A/(beta_A + beta_B)
s1 = beta_A*beta_B/(beta_A + beta_B)**3

m1_a = m1.reshape(-1, 1)
s1_a = s1.reshape(-1, 1)
beta_Aa = beta_A.reshape(-1, 1)
beta_Ba = beta_B.reshape(-1, 1)

r = np.linspace(0, 1, 100)

s.beta.ppf(1e-6, m1_a, s1_a)

norm = s.norm.cdf(r, m1_a, np.sqrt(s1_a))
beta = s.beta.cdf(r, beta_Aa, beta_Ba)

d = np.abs(norm - beta)[:, 1:-1].sum(1)
d = np.einsum('ij,ij->i', norm, beta)

plt.figure(1); plt.clf()
plt.imshow(np.log(d.reshape(99, -1))/np.log(10))
plt.colorbar()

# 5 point test
pts = np.linspace(s.beta.ppf(1e-10, beta_Aa, beta_Ba), s.beta.isf(1e-10, beta_Aa, beta_Ba), 10).squeeze().T

d = np.abs(s.norm.logpdf(pts, m1_a, np.sqrt(s1_a)) - s.beta.logpdf(pts, beta_Aa, beta_Ba)).sum(1)

plt.figure(2); plt.clf()
plt.imshow(np.log(d.reshape(beta_A.shape[0], -1))/np.log(10), extent = [1, 9951, 1, 9951], origin = "lower")
plt.imshow(d.reshape(beta_A.shape[0], -1))
plt.colorbar()

def score(a, b):
    pts = np.linspace(s.beta.ppf(1e-10, a, b), s.beta.isf(1e-10, a, b), 10).squeeze().T

    m1 = a/(a + b)
    s1 = a*b/(a + b)**3
    return np.abs(s.norm.logpdf(pts, m1, np.sqrt(s1)) - s.beta.logpdf(pts, a, b)).sum()
