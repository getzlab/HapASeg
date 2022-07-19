import numpy as np
import scipy.special as ss
import copy

class PoissonRegression:
    def __init__(self, r, C, Pi):
        self.r = r
        self.C = C
        self.Pi = Pi

        self.mu = np.log(r.mean() * np.ones([Pi.shape[1], 1]))
        self.beta = np.ones([C.shape[1], 1])
        self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu)

    # mu gradient
    def gradmu(self):
        return self.Pi.T @ (self.r - self.e_s)

    # mu Hessian
    def hessmu(self):
        return (-self.Pi.T * self.e_s.T)  @ self.Pi

    # beta gradient
    def gradbeta(self):
        return self.C.T @ (self.r - self.e_s)

    # beta Hessian
    def hessbeta(self):
        return (-self.C.T * self.e_s.T) @ self.C

    # mu,beta Hessian
    def hessmubeta(self):
        return (-self.C.T * self.e_s.T) @ self.Pi

    def NR_poisson(self):
        for i in range(100):
            self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu)
            gmu = self.gradmu()
            gbeta = self.gradbeta()
            grad = np.r_[gmu, gbeta]

            hmu = self.hessmu()
            hbeta = self.hessbeta()
            hmubeta = self.hessmubeta()
            H = np.r_[np.c_[hmu, hmubeta.T], np.c_[hmubeta, hbeta]]

            delta = np.linalg.inv(H) @ grad
            self.mu -= delta[0:len(self.mu)]
            self.beta -= delta[len(self.mu):]

            if np.linalg.norm(grad) < 1e-5:
                break

    def fit(self):
        self.NR_poisson()
        return self.mu, self.beta

hr, hw = ss.roots_legendre(25)
hr_extra, hw_extra = ss.roots_legendre(2500)

class CovLNP_NR:
    def __init__(self, x, beta, C, exposure=np.array([[0]]), extra_roots=False):
        """
        find posterior predictive over MCMC chains
        """
        #
        # legrende roots/weights for quadrature
        self.x =x
        self.beta = beta
        self.bc = C@beta
        self.bce = self.bc + exposure
        
        if extra_roots:
                self.hr, self.hw = hr_extra, hw_extra
        else:
                self.hr, self.hw = hr, hw
        
        # make empirical estimates about mu and sigma
        self.mu = (np.log(x) - self.bce).mean()
        self.lgsigma = np.log((np.log(x) - self.bce).std())

    def integral(self, x):
        """ Approximate marginalizing out latent parameter via Hermite quadrature.
            Assumes x is n x 1, c is n x n_c
        """
        I = self.hs*x - np.exp(self.bce)*self.hs_m_exp - self.h_roots**2 / 2
        m = np.max(I, 1, keepdims = True)

        return  m + np.log((self.b-self.a)/2 * np.exp(I - m)@self.h_weights)
    
    def lnp_logprob(self, x):
        """
        Compute LNP log PMF 
         x : counts
         c : covariate vector
         e : log exposures
        """

        return -ss.loggamma(x + 1) - np.log(2*np.pi)/2 + x*(self.bce + self.mu) + self.integral(x)
    
    def plot_fit(self):
        """
        plot LNP PMF over the data distribution
         x : counts
        """
        x = self.x
        log_prob_data = self.lnp_logprob(x).sum()
        
        # remove covariate effects
        x = np.exp(np.log(x) - self.bc)
        pmf_object = copy.deepcopy(self)

        step = int(np.ceil(x.size / 1000))
        xs= np.r_[max(x.min() - 20 * step, 0.1):x.max() + 20 * step:step][:, None]
        pmf_object.bce = np.zeros(xs.shape)
        interval_center = np.exp(pmf_object.lgsigma) * np.c_[xs] - np.exp(-pmf_object.lgsigma) * ss.wrightomega(pmf_object.mu + pmf_object.bce + 2 * pmf_object.lgsigma + np.exp(2*pmf_object.lgsigma) * np.c_[xs]).real
        epsi_hess = - np.exp(pmf_object.bce + pmf_object.mu + np.exp(pmf_object.lgsigma) * interval_center) * np.exp(2*pmf_object.lgsigma) -1
        epsi_sigma = np.sqrt(-epsi_hess**-1)
        interval_radius = 6 * epsi_sigma
        pmf_object.a, pmf_object.b = interval_center - interval_radius, interval_center + interval_radius
        pmf_object.h_roots = (pmf_object.b - pmf_object.a)/2 * pmf_object.hr[None] + (pmf_object.a + pmf_object.b)/ 2
        # cache intermediate values
        pmf_object.hs = pmf_object.h_roots*(np.exp(pmf_object.lgsigma))
        pmf_object.hs_m = pmf_object.hs + pmf_object.mu
        pmf_object.hs_m_exp = np.exp(pmf_object.hs + pmf_object.mu)
        pmf = pmf_object.lnp_logprob(xs)

        import matplotlib.pyplot as plt
        plt.hist(x.flatten(), bins = 100, density=True)
        plt.plot(xs.flatten(), np.exp(pmf.flatten()))

        plt.title('lnp fit: mu: {}  sigma {} ll: {}'.format(np.around(pmf_object.mu, 2), np.around(np.exp(pmf_object.lgsigma),2), np.around(log_prob_data,2)))
        plt.xlabel('corrected coverage')
        plt.ylabel('density')
        del pmf_object
    
    def gradhess(self):
        x = self.x
		## grad mu
        E = np.exp(self.bce)*self.hs_m_exp
        I_B = self.hs * x - E - self.h_roots**2 / 2
        I_A = self.hs_m + self.bce + I_B
        
        m = np.max(I_B, 1, keepdims = True) 
        m_A = np.max(I_A, 1, keepdims = True)
        
        A = (m_A + np.log(np.exp(I_A - m_A)@self.h_weights))
        B = (m + np.log(np.exp(I_B - m)@self.h_weights))
        
        gradmu = (x - np.exp(A - B)).sum()
    
        ## grad sigma 
        S_coef = self.hs * (x - E)
        # these coefficients can be negative so we save their sign and make positive
        S_coef_msk =  S_coef > 0
        S_coef_msk = (S_coef_msk * 2 - 1)
        I_S = np.log(S_coef * S_coef_msk) + I_B
        
        m_S = np.max(I_S, 1, keepdims = True)
        
        # we apply the sign correction to get the real, normalized results, but we still
        # need to multiply by the scale factor that we removed
        
        S_real = (np.exp(I_S - m_S)*S_coef_msk)@self.h_weights
        
        # to add back the log term we need to save the signs again
        S_real_msk = S_real > 0
        S_real_msk = S_real_msk * 2 - 1
        S = (m_S + np.log(S_real*S_real_msk))
        
        gradsigma = (S_real_msk * np.exp(S - B)).sum()
    
        ## hess mu mu
        # I_A term here techinically has a leading negative but we only use its square
        
        partial_A_coef = E - 1
        partial_A_coef_msk =  partial_A_coef > 0
        partial_A_coef_msk = (partial_A_coef_msk * 2 - 1)
        
        I_partial_A = np.log(partial_A_coef * partial_A_coef_msk) + self.hs_m + self.bce + I_B
        
        m_partial_A = np.max(I_partial_A, 1, keepdims = True)
        
        partial_A_real = (np.exp(I_partial_A - m_partial_A)*partial_A_coef_msk)@self.h_weights
        partial_A_real_msk = partial_A_real > 0
        partial_A_real_msk = (partial_A_real_msk * 2 - 1)
        partial_A = (m_partial_A + np.log(partial_A_real * partial_A_real_msk))
        
        hess_mu_mu = (partial_A_real_msk * np.exp(partial_A - B) - np.exp(2 * A - 2 * B)).sum()
        
        ## hess mu sigma       
        # I_A term is negative here, but we reflect that in the final division

        partial_A2_coef = - self.hs * E * (1 + x - E)
        partial_A2_coef_msk =  partial_A2_coef > 0
        partial_A2_coef_msk = (partial_A2_coef_msk * 2 - 1)

        I_partial_A2 = np.log(partial_A2_coef * partial_A2_coef_msk) + I_B

        m_partial_A2 = np.max(I_partial_A2, 1, keepdims = True)

        partial_A2_real = (np.exp(I_partial_A2 - m_partial_A2)*partial_A2_coef_msk)@self.h_weights
        partial_A2_real_msk = partial_A2_real > 0
        partial_A2_real_msk = (partial_A2_real_msk * 2 - 1)
        partial_A2 = (m_partial_A2 + np.log(partial_A2_real * partial_A2_real_msk))
        
        # the leading negative in I_A is reflected in the addtion of the second term
        hess_mu_sigma = (partial_A2_real_msk * np.exp(partial_A2 - B) + S_real_msk * np.exp(A + S - 2*B)).sum()
        
        ## hess sigma_sigma

        partial_S_coef = self.hs * (x + x * self.hs * (x - E) - E - self.hs * E - self.hs * E * (x - E))
        partial_S_coef_msk =  partial_S_coef > 0
        partial_S_coef_msk = (partial_S_coef_msk * 2 - 1)
        I_partial_S = np.log(partial_S_coef * partial_S_coef_msk) + I_B

        m_partial_S = np.max(I_partial_S, 1, keepdims = True)

        partial_S_real = (np.exp(I_partial_S - m_partial_S)*partial_S_coef_msk)@self.h_weights
        partial_S_real_msk = partial_S_real > 0
        partial_S_real_msk = (partial_S_real_msk * 2 - 1)
        partial_S = (m_partial_S + np.log(partial_S_real * partial_S_real_msk))

        #dont need to correct S sign since we square it
        hess_sigma_sigma = (partial_S_real_msk * np.exp(partial_S - B) - np.exp(2 * S - 2 * B)).sum()
        
        grad = np.array([gradmu, gradsigma])
        hess = np.array([[hess_mu_mu, hess_mu_sigma],
                        [hess_mu_sigma, hess_sigma_sigma]])
        return grad, hess
    
    def fit(self, ret_hess = False, debug=False, extra_roots=False):
        for _ in range(200):
            if debug: print(self.mu, self.lgsigma)
            radius_mult = 500 if extra_roots else 6
            x = self.x
            interval_center = np.exp(self.lgsigma) * x - np.exp(-self.lgsigma) * ss.wrightomega(self.mu + self.bce + 2 * self.lgsigma + np.exp(2*self.lgsigma) * x).real
            epsi_hess = - np.exp(self.bce + self.mu + np.exp(self.lgsigma) * interval_center) * np.exp(2*self.lgsigma) -1
            epsi_sigma = np.sqrt(-epsi_hess**-1)
            interval_radius = radius_mult * epsi_sigma
            
            self.a, self.b = interval_center - interval_radius, interval_center + interval_radius
            self.h_roots = (self.b - self.a)/2 * self.hr[None] + (self.a + self.b) / 2
            self.h_weights = np.c_[self.hw]
            # cache intermediate values
            self.hs = self.h_roots*(np.exp(self.lgsigma))
            self.hs_m = self.hs + self.mu
            self.hs_m_exp = np.exp(self.hs + self.mu)
            
            grad, hess = self.gradhess()
            
            if debug: print('grad:', grad)
            if debug: print('hess:', hess)
            delta = np.linalg.inv(hess) @ grad
            self.mu -= delta[0]
            self.lgsigma -= delta[1]
            if debug: print('grad_norm:', np.linalg.norm(grad))
            if np.linalg.norm(grad) < 5e-5:
                if ret_hess: return self.mu, self.lgsigma, hess
                return self.mu, self.lgsigma
        print('did not converge!')
        raise ValueError("DNC")
        if ret_hess: return None, None, None
        return None, None

# stand alone function for computing log likelihood of a segment under lnp model
def covLNP_ll(x, mu, lgsigma, C, beta, exposure=np.array([[0]])):
    #mu and lgsigma can either be doubles or nx1 arrays
    bce = C@beta + exposure
    
    interval_center = np.exp(lgsigma) * x - np.exp(-lgsigma) * ss.wrightomega(mu + bce + 2 * lgsigma + np.exp(2*lgsigma) * x).real
    epsi_hess = - np.exp(bce + mu + np.exp(lgsigma) * interval_center) * np.exp(2*lgsigma) -1
    epsi_sigma = np.sqrt(-epsi_hess**-1)
    interval_radius = 6 * epsi_sigma

    a, b = interval_center - interval_radius, interval_center + interval_radius
    h_roots = (b - a)/2 * hr[None] + (a + b) / 2
    h_weights = np.c_[hw]
    # cache intermediate values
    hs = h_roots*(np.exp(lgsigma))
    hs_m = hs + mu
    hs_m_exp = np.exp(hs + mu)
            
    I = hs*x - np.exp(bce)*hs_m_exp - h_roots**2 / 2
    m = np.max(I, 1, keepdims = True)

    Int =  m + np.log((b-a)/2 * np.exp(I - m)@h_weights)
    return (-ss.loggamma(x + 1) - np.log(2*np.pi)/2 + x*(bce + mu) + Int).sum()

# with prior
class CovLNP_NR_prior:
    def __init__(self, x, beta, C, exposure=np.array([[0]]), extra_roots=False, *, mu_prior, lamda, alpha_prior, beta_prior):
        """
        find posterior predictive over MCMC chains
        """
        # legrende roots/weights for quadrature
        self.x =x
        self.beta = beta
        self.bce = C@beta + exposure
        
        self.mu_prior = mu_prior
        self.lamda = lamda
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        if extra_roots:
                self.hr, self.hw = hr_extra, hw_extra
        else:
                self.hr, self.hw = hr, hw
        
        #make empirical estimates about mu and sigma
        self.mu = (np.log(x) - self.bce).mean()
        #self.lgsigma = np.log((np.log(x) - self.bce).std())
        # use sigma prior as starting point
        self.lgsigma = np.log(beta_prior / (alpha_prior + 1 + 0.5)) / 2
        self.sigma = np.exp(self.lgsigma)

    def integral(self, x):
        """ Approximate marginalizing out latent parameter via Hermite quadrature.
            Assumes x is n x 1, c is n x n_c
        """
        I = self.hs*x - np.exp(self.bce)*self.hs_m_exp - self.h_roots**2 / 2
        m = np.max(I, 1, keepdims = True)

        return  m + np.log((self.b-self.a)/2 * np.exp(I - m)@self.h_weights)
        
    def lnp_logprob(self, x):
        """
        Compute LNP log PMF 
         x : counts
         c : covariate vector
         e : log exposures
        """

        return -ss.loggamma(x + 1) - np.log(2*np.pi)/2 + x * (self.bce + self.mu) + self.integral(x) + np.log(self.lamda) / 2 - self.lgsigma + np.log(np.sqrt(2 * np.pi)) + self.alpha_prior * np.log(self.beta_prior) - ss.loggamma(self.alpha_prior) - (self.alpha_prior +1) * 2 * self.lgsigma - (2 * self.beta_prior + self.lamda * (self.mu - self.mu_prior)**2) / (2 * np.exp(self.lgsigma) **2)
    
    def plot_fit(self):
        """
        plot LNP PMF over the data distribution
         x : counts
        """
        x = self.x
        log_prob_data = self.lnp_logprob(x).sum()

        pmf_object = copy.deepcopy(self)

        x_corr = np.exp(np.log(x) - pmf_object.bce).flatten()
        #temporarily zero out bce
        pmf_object.bce = np.zeros((1000,1))
        xs= np.r_[0.1:1000:1]
        interval_center = np.exp(pmf_object.lgsigma) * np.c_[xs] - np.exp(-pmf_object.lgsigma) * ss.wrightomega(pmf_object.mu + pmf_object.bce + 2 * pmf_object.lgsigma + np.exp(2*pmf_object.lgsigma) * np.c_[xs]).real
        epsi_hess = - np.exp(pmf_object.bce + pmf_object.mu + np.exp(pmf_object.lgsigma) * interval_center) * np.exp(2*pmf_object.lgsigma) -1
        epsi_sigma = np.sqrt(-epsi_hess**-1)
        interval_radius = 6 * epsi_sigma
        pmf_object.a, pmf_object.b = interval_center - interval_radius, interval_center + interval_radius
        pmf_object.h_roots = (pmf_object.b - pmf_object.a)/2 * pmf_object.hr[None] + (pmf_object.a + pmf_object.b)/ 2
        # cache intermediate values
        pmf_object.hs = pmf_object.h_roots*(np.exp(pmf_object.lgsigma))
        pmf_object.hs_m = pmf_object.hs + pmf_object.mu
        pmf_object.hs_m_exp = np.exp(pmf_object.hs + pmf_object.mu)
        pmf = pmf_object.lnp_logprob(xs[:,None])

        import matplotlib.pyplot as plt
        plt.hist(x_corr.flatten(), bins = xs, density=True)
        plt.plot(xs, np.exp(pmf.flatten()))

        sig_bins = xs[pmf.flatten() > -30]
        plt.xlim([min(x_corr.min(), sig_bins[0]), max(x_corr.max(), sig_bins[-1])])
        plt.title('lnp fit: mu: {}  sigma {} ll: {}'.format(np.around(pmf_object.mu, 2), np.around(np.exp(pmf_object.lgsigma),2), np.around(log_prob_data,2)))
        plt.xlabel('corrected coverage')
        plt.ylabel('density')
        del pmf_object
    
    def gradhess(self):
        self.sigma = np.exp(self.lgsigma)
        x = self.x
        ## grad mu
        E = np.exp(self.bce)*self.hs_m_exp
        I_B = self.hs * x - E - self.h_roots**2 / 2
        I_A = self.hs_m + self.bce + I_B
        
        m = np.max(I_B, 1, keepdims = True) 
        m_A = np.max(I_A, 1, keepdims = True)
        
        A = (m_A + np.log(np.exp(I_A - m_A)@self.h_weights))
        B = (m + np.log(np.exp(I_B - m)@self.h_weights))
        
        gradmu = (x - np.exp(A - B)).sum() - self.lamda * (self.mu - self.mu_prior) / self.sigma**2
    
        ## grad sigma 
        S_coef = self.hs * (x - E)
        # these coefficients can be negative so we save their sign and make positive
        S_coef_msk =  S_coef > 0
        S_coef_msk = (S_coef_msk * 2 - 1)
        I_S = np.log(S_coef * S_coef_msk) + I_B
        
        m_S = np.max(I_S, 1, keepdims = True)
        
        # we apply the sign correction to get the real, normalized results, but we still
        # need to multiply by the scale factor that we removed
        
        S_real = (np.exp(I_S - m_S)*S_coef_msk)@self.h_weights
        
        # to add back the log term we need to save the signs again
        S_real_msk = S_real > 0
        S_real_msk = S_real_msk * 2 - 1
        S = (m_S + np.log(S_real*S_real_msk))
        
        gradsigma = (S_real_msk * np.exp(S - B)).sum() - (2*self.alpha_prior + 3) / self.sigma + (2 * self.beta_prior + self.lamda * (self.mu - self.mu_prior)**2) / self.sigma**3
    
        ## hess mu mu
        # I_A term here techinically has a leading negative but we only use its square
        
        partial_A_coef = E - 1
        partial_A_coef_msk =  partial_A_coef > 0
        partial_A_coef_msk = (partial_A_coef_msk * 2 - 1)
        
        I_partial_A = np.log(partial_A_coef * partial_A_coef_msk) + self.hs_m + self.bce + I_B
        
        m_partial_A = np.max(I_partial_A, 1, keepdims = True)
        
        partial_A_real = (np.exp(I_partial_A - m_partial_A)*partial_A_coef_msk)@self.h_weights
        partial_A_real_msk = partial_A_real > 0
        partial_A_real_msk = (partial_A_real_msk * 2 - 1)
        partial_A = (m_partial_A + np.log(partial_A_real * partial_A_real_msk))
        
        hess_mu_mu = (partial_A_real_msk * np.exp(partial_A - B) - np.exp(2 * A - 2 * B)).sum() - self.lamda / self.sigma**2
        
        ## hess mu sigma       
        # I_A term is negative here, but we reflect that in the final division

        partial_A2_coef = - self.hs * E * (1 + x - E)
        partial_A2_coef_msk =  partial_A2_coef > 0
        partial_A2_coef_msk = (partial_A2_coef_msk * 2 - 1)

        I_partial_A2 = np.log(partial_A2_coef * partial_A2_coef_msk) + I_B

        m_partial_A2 = np.max(I_partial_A2, 1, keepdims = True)

        partial_A2_real = (np.exp(I_partial_A2 - m_partial_A2)*partial_A2_coef_msk)@self.h_weights
        partial_A2_real_msk = partial_A2_real > 0
        partial_A2_real_msk = (partial_A2_real_msk * 2 - 1)
        partial_A2 = (m_partial_A2 + np.log(partial_A2_real * partial_A2_real_msk))
        
        # the leading negative in I_A is reflected in the addtion of the second term
        hess_mu_sigma = (partial_A2_real_msk * np.exp(partial_A2 - B) + S_real_msk * np.exp(A + S - 2*B)).sum() + 2 * self.lamda * (self.mu - self.mu_prior) / self.sigma**3
        
        ## hess sigma_sigma

        partial_S_coef = self.hs * (x + x * self.hs * (x - E) - E - self.hs * E - self.hs * E * (x - E))
        partial_S_coef_msk =  partial_S_coef > 0
        partial_S_coef_msk = (partial_S_coef_msk * 2 - 1)
        I_partial_S = np.log(partial_S_coef * partial_S_coef_msk) + I_B

        m_partial_S = np.max(I_partial_S, 1, keepdims = True)

        partial_S_real = (np.exp(I_partial_S - m_partial_S)*partial_S_coef_msk)@self.h_weights
        partial_S_real_msk = partial_S_real > 0
        partial_S_real_msk = (partial_S_real_msk * 2 - 1)
        partial_S = (m_partial_S + np.log(partial_S_real * partial_S_real_msk))

        #dont need to correct S sign since we square it
        hess_sigma_sigma = (partial_S_real_msk * np.exp(partial_S - B) - np.exp(2 * S - 2 * B)).sum() + (2 * self.alpha_prior + 3) / self.sigma**2 - 3 * (2 * self.beta_prior + self.lamda * (self.mu - self.mu_prior)**2) / self.sigma**4
        
        grad = np.array([gradmu, gradsigma])
        hess = np.array([[hess_mu_mu, hess_mu_sigma],
                        [hess_mu_sigma, hess_sigma_sigma]])
        return grad, hess
    
    def fit(self, ret_hess = False, debug=False, extra_roots=False):
        radius_mult = 500 if extra_roots else 6
        for it in range(200):
            if debug: print(self.mu, self.lgsigma)
            x = self.x
            interval_center = np.exp(self.lgsigma) * x - np.exp(-self.lgsigma) * ss.wrightomega(self.mu + self.bce + 2 * self.lgsigma + np.exp(2*self.lgsigma) * x).real
            epsi_hess = - np.exp(self.bce + self.mu + np.exp(self.lgsigma) * interval_center) * np.exp(2*self.lgsigma) -1
            epsi_sigma = np.sqrt(-epsi_hess**-1)
            interval_radius = radius_mult * epsi_sigma
            
            self.a, self.b = interval_center - interval_radius, interval_center + interval_radius
            self.h_roots = (self.b - self.a)/2 * self.hr[None] + (self.a + self.b) / 2
            self.h_weights = np.c_[self.hw]
            # cache intermediate values
            self.hs = self.h_roots*(np.exp(self.lgsigma))
            self.hs_m = self.hs + self.mu
            self.hs_m_exp = np.exp(self.hs + self.mu)
            
            grad, hess = self.gradhess()
            if debug: print('grad:', grad)
            if debug: print('hess:', hess)
            delta = np.linalg.inv(hess) @ grad
            self.mu -= delta[0]
            self.lgsigma -= delta[1]
            if debug: print('grad_norm:', np.linalg.norm(grad))
            if np.linalg.norm(grad) < 1e-5:
                if it > 100: print('took {} iterations'.format(it))
                if ret_hess: return self.mu, self.lgsigma, hess
                return self.mu, self.lgsigma
        print('did not converge!')
        raise ValueError("DNC")
        if ret_hess: return None, None, None
        return None, None

def covLNP_ll_prior(x, mu, lgsigma, C, beta, exposure=np.array([[0]]), *, mu_prior, lamda, alpha_prior, beta_prior):
    #mu and lgsigma can either be doubles or nx1 arrays
    bce = C@beta + exposure
    
    interval_center = np.exp(lgsigma) * x - np.exp(-lgsigma) * ss.wrightomega(mu + bce + 2 * lgsigma + np.exp(2*lgsigma) * x).real
    epsi_hess = - np.exp(bce + mu + np.exp(lgsigma) * interval_center) * np.exp(2*lgsigma) -1
    epsi_sigma = np.sqrt(-epsi_hess**-1)
    interval_radius = 6 * epsi_sigma

    a, b = interval_center - interval_radius, interval_center + interval_radius
    h_roots = (b - a)/2 * hr[None] + (a + b) / 2
    h_weights = np.c_[hw]
    # cache intermediate values
    hs = h_roots*(np.exp(lgsigma))
    hs_m = hs + mu
    hs_m_exp = np.exp(hs + mu)
            
    I = hs*x - np.exp(bce)*hs_m_exp - h_roots**2 / 2
    m = np.max(I, 1, keepdims = True)

    Int =  m + np.log((b-a)/2 * np.exp(I - m)@h_weights)
    sigma = np.exp(lgsigma)
    return (-ss.loggamma(x + 1) - np.log(2*np.pi)/2 + x*(bce + mu) + Int).sum() + np.log(lamda) / 2 - lgsigma + np.log(np.sqrt(2 * np.pi)) + alpha_prior * np.log(beta_prior) - ss.loggamma(alpha_prior) - (alpha_prior +1) * 2 * lgsigma - (2 * beta_prior + lamda * (mu - mu_prior)**2) / (2 * np.exp(lgsigma) **2)
