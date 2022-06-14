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

class CovLNP_NR:
    def __init__(self, x, beta, C, exposure=np.array([[0]])):
        """
        find posterior predictive over MCMC chains
        """
        #
        # legrende roots/weights for quadrature
        self.x =x
        self.beta = beta
        self.bce = C@beta + exposure

        self.hr, self.hw = ss.roots_legendre(25)
        
        #make empirical estimates about mu and sigma
        self.mu = (np.log(x) - self.bce).mean()
        self.lgsigma = np.log((np.log(x) - self.bce).std())

    def integral(self, x):
        """ Approximate marginalizing out latent parameter via Hermite quadrature.
            Assumes x is n x 1, c is n x n_c
        """
        I = self.hs*x - np.exp(self.bce)*self.hs_m_exp - self.h_roots**2 / 2
        m = np.max(I, 1, keepdims = True)

        return  m + np.log((self.b-self.a)/2 * np.exp(I - m)@self.h_weights)
    
    def gradmu(self, x):
        E = np.exp(self.bce)*self.hs_m_exp
        I_base = self.hs * x - E - self.h_roots**2 / 2
        I_grad_mu = self.hs_m + self.bce + I_base
        
        m = np.max(I_base, 1, keepdims = True) 
        m_grad_mu = np.max(I_grad_mu, 1, keepdims = True)
        
        num_sum = (m_grad_mu + np.log(np.exp(I_grad_mu - m_grad_mu)@self.h_weights))
        denom_sum = (m + np.log(np.exp(I_base - m)@self.h_weights))
        
        return (x - np.exp(num_sum - denom_sum)).sum()
    
    def gradsigma(self,x):
        E = np.exp(self.bce)*self.hs_m_exp
        I_base = self.hs * x - E - self.h_roots**2 / 2
        coef = self.hs * ( x - E)
        # these coefficients can be negative so we save their sign and make positive
        coef_msk =  coef > 0
        coef_msk = (coef_msk * 2 - 1)
        I_grad = np.log(coef * coef_msk) + I_base
        
        m_grad = np.max(I_grad, 1, keepdims = True)
        m = np.max(I_base, 1, keepdims = True) 
        
        # we apply the sign correction to get the real, normalized results, but we still
        # need to multiply by the scale factor that we removed
        
        small_grad = (np.exp(I_grad - m_grad)*coef_msk)@self.h_weights
        
        # to add back the log term we need to save the signs again
        small_msk = small_grad > 0
        small_msk = small_msk * 2 - 1
        num_sum = (m_grad + np.log(small_grad*small_msk))

        denom_sum = (m + np.log(np.exp(I_base - m)@self.h_weights))
        
        return (small_msk * np.exp(num_sum - denom_sum)).sum()
    
    def hess_mu_mu(self, x):
        E = np.exp(self.bce)*self.hs_m_exp
        I_base = self.hs*x - E - self.h_roots**2 / 2
        I_A = self.hs_m + self.bce + I_base # this has a leading negative but we only use its square
        
        partial_A_coef = E - 1
        partial_A_coef_msk =  partial_A_coef > 0
        partial_A_coef_msk = (partial_A_coef_msk * 2 - 1)
        
        I_partial_A = np.log(partial_A_coef * partial_A_coef_msk) + self.hs_m + self.bce + I_base
        
        m = np.max(I_base, 1, keepdims = True) 
        m_A = np.max(I_A, 1, keepdims = True)
        m_partial_A = np.max(I_partial_A, 1, keepdims = True)
        
        partial_A = (np.exp(I_partial_A - m_partial_A)*partial_A_coef_msk)@self.h_weights
        partial_A_msk = partial_A > 0
        partial_A_msk = (partial_A_msk * 2 - 1)
        partial_A = (m_partial_A + np.log(partial_A * partial_A_msk))
                     
        A = (m_A + np.log(np.exp(I_A - m_A)@self.h_weights))
        B = (m + np.log(np.exp(I_base - m)@self.h_weights))
        
        return (partial_A_msk * np.exp(partial_A - B) - np.exp(2 * A - 2 * B)).sum()
    
    def hess_mu_sigma(self, x):
        E = np.exp(self.bce)*self.hs_m_exp
        I_base = self.hs * x - E - self.h_roots**2 / 2
        # negative
        I_A = self.hs_m + self.bce + I_base

        partial_A_coef = - self.hs * E * (1 + x - E)
        partial_A_coef_msk =  partial_A_coef > 0
        partial_A_coef_msk = (partial_A_coef_msk * 2 - 1)

        I_partial_A = np.log(partial_A_coef * partial_A_coef_msk) + I_base

        partial_B_coef = self.hs * (x - E)
        partial_B_coef_msk =  partial_B_coef > 0
        partial_B_coef_msk = (partial_B_coef_msk * 2 - 1)

        I_partial_B = np.log(partial_B_coef * partial_B_coef_msk) + I_base

        m = np.max(I_base, 1, keepdims = True) 
        m_A = np.max(I_A, 1, keepdims = True)
        m_partial_A = np.max(I_partial_B, 1, keepdims = True)
        m_partial_B = np.max(I_partial_B, 1, keepdims = True)

        partial_A = (np.exp(I_partial_A - m_partial_A)*partial_A_coef_msk)@self.h_weights
        partial_A_msk = partial_A > 0
        partial_A_msk = (partial_A_msk * 2 - 1)
        partial_A = (m_partial_A + np.log(partial_A * partial_A_msk))

        partial_B = (np.exp(I_partial_B - m_partial_B)*partial_B_coef_msk)@self.h_weights
        partial_B_msk = partial_B > 0
        partial_B_msk = (partial_B_msk * 2 - 1)
        partial_B = (m_partial_B + np.log(partial_B * partial_B_msk))

        A = (m_A + np.log(np.exp(I_A - m_A)@self.h_weights))
        B = (m + np.log(np.exp(I_base - m)@self.h_weights))
        
        # the leading negative in I_A is reflected in the addtion of the second term
        return (partial_A_msk * np.exp(partial_A - B) + partial_B_msk * np.exp(A + partial_B - 2*B)).sum()
    
    def hess_sigma_sigma(self, x):
        E = np.exp(self.bce)*self.hs_m_exp
        I_base = self.hs*x - E - self.h_roots**2 / 2

        I_S_coef = self.hs * (x - E)
        I_S_coef_msk =  I_S_coef > 0
        I_S_coef_msk = (I_S_coef_msk * 2 - 1)
        I_S = np.log(I_S_coef * I_S_coef_msk) + I_base

        I_partial_S_coef = self.hs * (x + x * self.hs * (x - E) - E - self.hs * E - self.hs * E * (x - E))
        I_partial_S_coef_msk =  I_partial_S_coef > 0
        I_partial_S_coef_msk = (I_partial_S_coef_msk * 2 - 1)
        I_partial_S = np.log(I_partial_S_coef * I_partial_S_coef_msk) + I_base

        m = np.max(I_base, 1, keepdims = True)
        m_S = np.max(I_S, 1, keepdims = True)
        m_partial_S = np.max(I_partial_S, 1, keepdims = True)

        _S = (np.exp(I_S - m_S)*I_S_coef_msk)@self.h_weights
        S_msk = _S > 0
        S_msk = (S_msk * 2 - 1)
        S = (m_S + np.log(_S * S_msk))

        _partial_S = (np.exp(I_partial_S - m_partial_S)*I_partial_S_coef_msk)@self.h_weights
        partial_S_msk = _partial_S > 0
        partial_S_msk = (partial_S_msk * 2 - 1)
        partial_S = (m_partial_S + np.log(_partial_S * partial_S_msk))

        B = (m + np.log(np.exp(I_base - m)@self.h_weights))

        #dont need to correct S sign since we square it
        return (partial_S_msk * np.exp(partial_S - B) - np.exp(2 * S - 2 * B)).sum()
        
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
    
    def fit(self, ret_hess = False, debug=False):
        for _ in range(100):
            if debug: print(self.mu, self.lgsigma)
            x = self.x
            interval_center = np.exp(self.lgsigma) * x - np.exp(-self.lgsigma) * ss.wrightomega(self.mu + self.bce + 2 * self.lgsigma + np.exp(2*self.lgsigma) * x).real
            epsi_hess = - np.exp(self.bce + self.mu + np.exp(self.lgsigma) * interval_center) * np.exp(2*self.lgsigma) -1
            epsi_sigma = np.sqrt(-epsi_hess**-1)
            interval_radius = 6 * epsi_sigma
            
            self.a, self.b = interval_center - interval_radius, interval_center + interval_radius
            self.h_roots = (self.b - self.a)/2 * self.hr[None] + (self.a + self.b) / 2
            self.h_weights = np.c_[self.hw]
            # cache intermediate values
            self.hs = self.h_roots*(np.exp(self.lgsigma))
            self.hs_m = self.hs + self.mu
            self.hs_m_exp = np.exp(self.hs + self.mu)
            
            grad = np.array([self.gradmu(x), self.gradsigma(x)])
            hess = np.array([[self.hess_mu_mu(x), self.hess_mu_sigma(x)],
                            [self.hess_mu_sigma(x), self.hess_sigma_sigma(x)]])
            if debug: print('grad:', grad)
            if debug: print('hess:', hess)
            delta = np.linalg.inv(hess) @ grad
            self.mu -= delta[0]
            self.lgsigma -= delta[1]
            if debug: print('grad_norm:', np.linalg.norm(grad))
            if np.linalg.norm(grad) < 1e-5:
                if ret_hess: return self.mu, self.lgsigma, hess
                return self.mu, self.lgsigma
        print('did not converge!')
        if ret_hess: return None, None, None
        return None, None

# stand alone function for computing log likelyhood of a segment under lnp model
def covLNP_ll(x, mu, lgsigma, C, beta, exposure=np.array([[0]])):
    #mu and lgsigma can either be doubles or nx1 arrays
    bce = C@beta + exposure
    hr, hw = ss.roots_legendre(25)
    
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
    return -ss.loggamma(x + 1) - np.log(2*np.pi)/2 + x*(bce + mu) + Int
