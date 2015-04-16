import numpy as np

import scipy.signal


class SINMF(object):
    
    def __init__(self, n_bases, window_width, n_iter):
        
        self.n_iter = n_iter
        self.window_width = window_width
        self.n_bases = n_bases                
        
    def fit(self, X):
        
        N_timesteps, N_features = X.shape
        
        # Initialize in a super naive way
        A = np.random.random((self.n_bases, N_timesteps))+2
        D = np.random.random((self.n_bases, self.window_width, N_features))+2
        
        for _ in range(self.n_iter):
    
            self._update_activations(A, D, X)
            self._update_dictionary(A, D, X)
        
        return A, D
        
    def reconstruct(self, A, D):

        N_timesteps = A.shape[1]

        X_bar = np.zeros((N_timesteps, N_features))

        for basis, activation in zip(D, A):
            X_bar += scipy.signal.fftconvolve(basis.T, np.atleast_2d(activation)).T[:N_timesteps]

        return X_bar

    def _update_activations(self, A, D, X):
    
        for t_prime in range(self.window_width):

            X_bar = self.reconstruct(A, D)
            R = X/X_bar

            U_A = np.einsum("jk,tk->jt", D[:,t_prime,:]/np.atleast_2d(D[:,t_prime,:].sum(axis=1)).T, R[t_prime:]) 

            A[:,:-t_prime or None] *= U_A

    def _update_dictionary(self, A, D, X):
    
        for t_prime in range(self.window_width):

            X_bar = self.reconstruct(A, D)
            R = X/X_bar

            U_D = np.einsum("jn,ni->ji", A[:,:-t_prime or None]/np.atleast_2d(A[:,:-t_prime or None].sum(axis=1)).T, R[t_prime:])

            D[:,t_prime,:] *= U_D
