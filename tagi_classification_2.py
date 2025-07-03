import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

# Data generation and normalization
class Normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


def create_sin_data(N=200, noise_std=3):
    x = np.linspace(-5, 5, N)
    y = x**3 + np.random.randn(N) * noise_std
    return x, y

# Initialization routines
def init_parameters(units, prior_var=0.5):
    params = {'mw': [], 'Sw': [], 'mb': [], 'Sb': []}
    for in_dim, out_dim in zip(units[:-1], units[1:]):
        init_var = prior_var / in_dim
        params['mw'].append(np.random.randn(in_dim, out_dim) * np.sqrt(init_var))
        params['Sw'].append(np.full((in_dim, out_dim), init_var))
        params['mb'].append(np.zeros(out_dim))
        params['Sb'].append(np.full(out_dim, prior_var))
    return params


def init_states(units):
    states = {k: [np.zeros(dim) for dim in units] for k in ('mz','Sz','ma','Sa','J','cov_yz')}
    return states


def init_covariances(units):
    cov = {'cov_z_w': [], 'cov_z_z': [], 'cov_z_b': []}
    for in_dim, out_dim in zip(units[:-1], units[1:]):
        cov['cov_z_w'].append(np.zeros((in_dim, out_dim)))
        cov['cov_z_z'].append(np.zeros((in_dim, out_dim)))
        cov['cov_z_b'].append(np.zeros(out_dim))
    return cov

# Forward and backward passes
def feed_forward(x, units, params, states, cov):
    # Input layer
    states['ma'][0] = np.array([x]).flatten()
    states['Sa'][0] = np.zeros_like(states['ma'][0])

    # Layer propagation
    for i in range(len(units)-1):
        mw, Sw = params['mw'][i], params['Sw'][i]
        mb, Sb = params['mb'][i], params['Sb'][i]
        ma, Sa = states['ma'][i], states['Sa'][i]

        # pre-activation moments
        mz = ma @ mw + mb
        Sz = (ma**2) @ Sw + Sa @ (mw**2 + Sw) + Sb

        # store
        states['mz'][i+1], states['Sz'][i+1] = mz, Sz

        # activation
        if i == len(units)-2:
            ma_next, Sa_next = mz, Sz
            J = np.ones_like(mz)
        else:
            ma_next = relu(mz)
            J = drelu(mz)
            Sa_next = Sz * J

        states['ma'][i+1], states['Sa'][i+1], states['J'][i+1] = ma_next, Sa_next, J

        # covariances
        cov['cov_z_w'][i] = np.outer(ma, Sw.diagonal())
        cov['cov_z_b'][i] = Sb
        cov['cov_z_z'][i] = np.outer(Sa, J) * mw

        # output covariance of y on last layer
        if i == len(units)-2:
            states['cov_yz'][-1] = Sz

    return params, states, cov


def states_feed_backward(y, params, units, states, cov, sigma_v):
    # initialize deltas
    L = len(units)
    deltas = {k: [np.zeros(units[i]) for i in range(L)]
              for k in ('dmz','dSz','dmw','dSw','dmb','dSb')}

    # output layer delta
    my, Sy = states['mz'][-1][0], states['Sz'][-1][0] + sigma_v**2
    cov_yz = states['cov_yz'][-1][0]
    deltas['dmz'][-1][0] = my + cov_yz/Sy * (y - my)
    deltas['dSz'][-1][0] = states['Sz'][-1][0] - (cov_yz**2)/Sy

    # backwards through params
    for i in reversed(range(L-1)):
        ni, no = units[i], units[i+1]
        # weight and bias updates
        dmw = np.zeros_like(params['mw'][i])
        dSw = np.zeros_like(params['Sw'][i])
        dmb = np.zeros_like(params['mb'][i])
        dSb = np.zeros_like(params['Sb'][i])

        for j in range(ni):
            for k in range(no):
                Jw = cov['cov_z_w'][i][j,k] / states['Sz'][i+1][k]
                dmw[j,k] = params['mw'][i][j,k] + Jw*(deltas['dmz'][i+1][k] - states['mz'][i+1][k])
                dSw[j,k] = params['Sw'][i][j,k] + Jw*(deltas['dSz'][i+1][k] - states['Sz'][i+1][k])*Jw
            Jb = cov['cov_z_b'][i][j] / states['Sz'][i+1][j]
            dmb[j] = params['mb'][i][j] + Jb*(deltas['dmz'][i+1][j] - states['mz'][i+1][j])
            dSb[j] = params['Sb'][i][j] + Jb*(deltas['dSz'][i+1][j] - states['Sz'][i+1][j])*Jb

        deltas['dmw'][i], deltas['dSw'][i] = dmw, dSw
        deltas['dmb'][i], deltas['dSb'][i] = dmb, dSb

        # backward to hidden states
        if i > 0:
            for j in range(ni):
                aux_mz = aux_Sz = 0
                for k in range(no):
                    Jz = cov['cov_z_z'][i][j,k] / states['Sz'][i+1][k]
                    aux_mz += Jz*(deltas['dmz'][i+1][k] - states['mz'][i+1][k])
                    aux_Sz += Jz*(deltas['dSz'][i+1][k] - states['Sz'][i+1][k])*Jz
                deltas['dmz'][i][j] = states['mz'][i][j] + aux_mz
                deltas['dSz'][i][j] = states['Sz'][i][j] + aux_Sz

    # apply updates
    for i in range(L-1):
        params['mw'][i] = deltas['dmw'][i]
        params['Sw'][i] = deltas['dSw'][i]
        params['mb'][i] = deltas['dmb'][i]
        params['Sb'][i] = deltas['dSb'][i]
        states['mz'][i+1] = deltas['dmz'][i+1]
        states['Sz'][i+1] = deltas['dSz'][i+1]

    return params, states, cov

# Training and evaluation
def train_bnn(x_train, y_train, units, epochs, sigma_v):
    params = init_parameters(units)
    states = init_states(units)
    cov = init_covariances(units)
    for _ in range(epochs):
        idx = np.random.permutation(len(x_train))
        for i in idx:
            params, states, cov = feed_forward(x_train[i], units, params, states, cov)
            params, states, cov = states_feed_backward(y_train[i], params, units, states, cov, sigma_v)
    return params, states, cov


def predict_bnn(x, params, states, cov, units):
    m, S = feed_forward(x, units, params, states, cov)[1:3]
    return m, np.sqrt(S)

# Main script
def main():
    # data
    x, y = create_sin_data()
    N = len(x)
    idx = np.random.permutation(N)
    split = int(0.8*N)
    x_train, y_train = x[idx[:split]], y[idx[:split]]
    x_test, y_test = x[idx[split:]], y[idx[split:]]

    # normalize
    x_norm = Normal(x_train.mean(), x_train.std())
    y_norm = Normal(y_train.mean(), y_train.std())
    x_train = x_norm.normalize(x_train)
    y_train = y_norm.normalize(y_train)
    x_test = x_norm.normalize(x_test)
    y_test = y_norm.normalize(y_test)

    # BNN setup
    units = [1, 50, 1]
    sigma_v = 3 / y_norm.std
    params, states, cov = train_bnn(x_train, y_train, units, epochs=10, sigma_v=sigma_v)

    # prediction
    y_pred, sy_pred = [], []
    for xi in x_test:
        m, s = predict_bnn(xi, params, states, cov, units)
        y_pred.append(m)
        sy_pred.append(s)
    y_pred, sy_pred = np.array(y_pred), np.array(sy_pred)

    # evaluate and plot
    mse = np.mean((y_test - y_pred)**2)
    print(f'MSE: {mse:.4f}')

    # sort for plotting
    order = np.argsort(x_test)
    xt, yt, yp, sp = x_test[order], y_test[order], y_pred[order], sy_pred[order]

    plt.scatter(xt, yt, label='True')
    plt.scatter(xt, yp, label='Pred')
    plt.fill_between(xt, yp - sp, yp + sp, alpha=0.3)
    plt.title('Bayesian NN Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
