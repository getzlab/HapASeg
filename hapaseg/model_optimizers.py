import numpy as np


class PoissonRegression:
    def __init__(self, r, C, Pi,
      log_exposure = 0, log_offset = 0, intercept = True,
      mumu = 0, musig2 = 10, betamu = None, betasiginv = None):
        self.r = r
        self.C = C
        self.Pi = Pi
        self.log_exposure = log_exposure
        self.log_offset = log_offset
        self.intercept = intercept

        self.mu = np.log(r.mean() * np.ones([Pi.shape[1], 1])) - self.log_exposure
        self.beta = np.zeros([C.shape[1], 1])
        self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu + self.log_exposure + self.log_offset)

        # prior parameters
        self.mumu = mumu
        self.musig2 = musig2
        self.betamu = np.zeros_like(self.beta) if betamu is None else betamu
        self.betasiginv = 1/np.sqrt(10)*np.eye(len(self.beta)) if betasiginv is None else betasiginv

    # mu gradient
    def gradmu(self):
        return self.Pi.T @ (self.r - self.e_s) - (self.mu - self.mumu)/self.musig2

    # mu Hessian
    def hessmu(self):
        return (-self.Pi.T * self.e_s.T)  @ self.Pi - 1/self.musig2

    # beta gradient
    def gradbeta(self):
        return self.C.T @ (self.r - self.e_s) - self.betasiginv@(self.beta - self.betamu)

    # beta Hessian
    def hessbeta(self):
        return (-self.C.T * self.e_s.T) @ self.C - self.betasiginv

    # mu,beta Hessian
    def hessmubeta(self):
        return (-self.C.T * self.e_s.T) @ self.Pi

    def NR_poisson(self):
        for i in range(100):
            self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu + self.log_exposure + self.log_offset)
            gbeta = self.gradbeta()
            if self.intercept:
                gmu = self.gradmu()
                grad = np.r_[gmu, gbeta]
            else:
                grad = gbeta

            hbeta = self.hessbeta()
            if self.intercept:
                hmubeta = self.hessmubeta()
                hmu = self.hessmu()
                H = np.r_[np.c_[hmu, hmubeta.T], np.c_[hmubeta, hbeta]]
            else:
                H = hbeta

            delta = np.linalg.inv(H) @ grad
            if self.intercept:
                self.mu -= delta[0:len(self.mu)]
                self.beta -= delta[len(self.mu):]
            else:
                self.beta -= delta

            if np.linalg.norm(grad) < 1e-5:
                break

    def fit(self):
        self.NR_poisson()
        if self.intercept:
            return self.mu, self.beta
        else:
            return self.beta

    def hess(self):
        hbeta = self.hessbeta()
        if self.intercept:
            hmu = self.hessmu()
            hmubeta = self.hessmubeta()
            return np.r_[np.c_[hmu, hmubeta.T], np.c_[hmubeta, hbeta]]
        else:
            return hbeta
