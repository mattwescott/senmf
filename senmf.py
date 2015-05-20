import numpy as np

import scipy.signal

class SENMF(object):
    def __init__(self, n_bases, window_width, X):
        self.n_bases = n_bases
        self.window_width = window_width
        self.n_timesteps, self.n_features = X.shape
        self.X = X
        self.A = None
        self.D = None
        self.R = None

    def rand_A(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.A = np.random.random((self.n_bases, self.n_timesteps))+2
        return self.A

    def rand_D(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.D = np.random.random((self.n_bases, self.window_width, self.n_features))+2
        return self.D

    def normalize_D(self):
        for i in range(self.n_bases):
            self.D[i] /= np.linalg.norm(self.D[i])

    def reconstruct(self):
        "Reconstruct an estimation of the training data"
        X_bar = np.zeros((self.n_timesteps, self.n_features))
        for basis, activation in zip(self.D, self.A):
            X_bar += scipy.signal.fftconvolve(basis.T, np.atleast_2d(activation)).T[:self.n_timesteps]
        return X_bar

    def reconstruct_basis(self, basis):
        return scipy.signal.fftconvolve(self.D[basis].T, np.atleast_2d(self.A[basis]))

    def residual(self):
        "calculate the multiplicative residual error"
        return self.X / np.abs(self.reconstruct())

    def update_residual(self):
        "calc and store residual for future use"
        self.R = self.residual()

    def update_A(self):
        "Using stored residual, calculate and apply an update to activations"
        for t_prime in range(self.window_width):
            U_A = np.einsum(
                    "jk,tk->jt",
                    self.D[:,t_prime,:]/np.atleast_2d(self.D[:,t_prime,:].sum(axis=1)).T,
                    self.R[t_prime:])
            self.A[:,:-t_prime or None] *= U_A

    def update_A_fast(self):
        "Using stored residual, calculate and apply an update to activations"
        for t_prime in range(self.window_width):
            self.update_residual()
            U_A = np.einsum(
                    "jk,tk->jt",
                    self.D[:,t_prime,:]/np.atleast_2d(self.D[:,t_prime,:].sum(axis=1)).T,
                    self.R[t_prime:])
            self.A[:,:-t_prime or None] *= U_A

    def D_delta(self):
        D_updates = np.zeros((self.n_bases, self.window_width, self.n_features))
        for t_prime in range(self.window_width):
            U_D = np.einsum("jn,ni->ji", self.A[:,:-t_prime or None]/np.atleast_2d(self.A[:,:-t_prime or None].sum(axis=1)).T, self.R[t_prime:])
            D_updates[:,t_prime,:] = U_D
        return D_updates

    def update_D(self):
        "Using stored residual, calculate and apply an update to dictionary"
        self.D *= self.D_delta()

    def fit(self, n_iter):
        for _ in range(n_iter):
            self.update_A_fast()
            self.update_residual()
            self.update_D()

