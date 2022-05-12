import numpy as np


class PoissonRegression:
    def __init__(self, r, C, Pi, log_exposure = 0):
        self.r = r
        self.C = C
        self.Pi = Pi
        self.log_exposure = log_exposure

        self.mu = np.log(r.mean() * np.ones([Pi.shape[1], 1]))
        self.beta = np.ones([C.shape[1], 1])
        self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu + self.log_exposure)

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
            self.e_s = np.exp(self.C @ self.beta + self.Pi @ self.mu + self.log_exposure)
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
