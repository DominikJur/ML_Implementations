import numpy as np


class FullyConnected:
    
    def __init__(self, N_in, N_out):
        self.N_in = N_in
        self.N_out = N_out
        self.std = np.sqrt(4/ (N_in + N_out)) # Xavier initialization
        self.B = np.random.normal(0, self.std, (N_out,))
        self.Omega = np.random.normal(0, self.std, (N_out, N_in))
        self.dB = np.zeros_like(self.B)
        self.dOmega = np.zeros_like(self.Omega)
        self.passes = 0
        
        
    def forward(self, h):
        self.h = h
        return self.B + np.dot(self.Omega, h)

    def backward(self, dL_df):
        dL_dB = dL_df
        dL_dOmega = np.outer(dL_df, self.h)

        self.dB += dL_dB
        self.dOmega += dL_dOmega
        self.passes+=1        

        return np.dot(self.Omega.T, dL_df)
    
    def step(self, eta):
        self.B -= eta * self.dB/self.passes
        self.Omega -= eta * self.dOmega/self.passes
        self.dB = np.zeros_like(self.B)
        self.dOmega = np.zeros_like(self.Omega)
        self.passes=0
        
        
class ReLU:

    def forward(self, h):
        self.h = h
        return np.maximum(0, h)
    
    def backward(self, dL_df):
        dL_dh = dL_df * (self.h > 0).astype(float)
        return dL_dh
    
    def step(self, eta):
        return None
    
def MSE(f, y):
    assert f.shape == y.shape
    return np.sum((f-y)**2)

def dMSE_df(f, y):
    assert f.shape == y.shape
    return 2*(f-y)
    

class NeuralNet:
    
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, X):
        X = X.copy()
        if X.ndim == 1: # Handle a single sample input
            X = X.reshape(1, -1)
        dim = X.shape[0]
        f_out = []
        for i in range(dim):
            x = X[i].squeeze()
            for layer in self.layers:
                x = layer.forward(x)
            f_out.append(x)
        return np.array(f_out).squeeze()

        
    def fit(self, X, y, epochs=100, eta=1e-3):
        X,y = X.copy(), y.copy()
        dim = X.shape[0]
        assert len(y) == dim
        for j in range(epochs):
            total_loss = 0
            
            for i in range(dim): # Gradient descent
                f_ = X[i].squeeze()
                y_ = y[i].squeeze()
                for layer in self.layers:
                    f_ = layer.forward(f_)

                L = MSE(f_, y_)
                total_loss += L / dim

                dL_df = dMSE_df(f_, y_)
                for layer in reversed(self.layers):
                    dL_df = layer.backward(dL_df)
                    
            for layer in self.layers:
                layer.step(eta)

            print(f"Epoch {j + 1}/{epochs}, Loss: {total_loss:.4f}")

    def __call__(self, X):
        return self.forward(X)
    
    
class Softmax:

    def forward(self, X):
        e_X = np.exp(X)
        self.out = e_X / e_X.sum(axis=0)
        return self.out

    def backward(self, dE_dy):
        dot_prod = np.sum(dE_dy * self.out, axis=0, keepdims=True) 
        dE_dh = self.out * (dE_dy - dot_prod)
        return dE_dh

    def step(self, eta):
        return None
    
class Dropout:

    def __init__(self, p):
        self.p = p

    def forward(self, X):
        self.mask = (np.random.rand(*X.shape) > self.p).astype(float)
        return X * self.mask

    def backward(self, dE_dy):
        return dE_dy * self.mask

    def step(self, eta):
        return None
    
class Flatten:

    def forward(self, X):
        self.input_shape = X.shape
        return X.flatten()

    def backward(self, dE_dy):
        return dE_dy.reshape(self.input_shape)

    def step(self, eta):
        return None