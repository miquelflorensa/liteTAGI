import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

class Normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

def ReLU(x):
    return np.maximum(0, x)

def create_sin_data():
    x = np.linspace(-5, 5, 200)
    y = x**3 + np.random.randn(len(x))*3
    return x, y

def create_sin_data_2D_input():
    # Example with 2 inputs and 1 output
    x = np.linspace(-5, 5, 200)
    x = np.column_stack((x, 0.5*x))  # 2 input features
    y = x[:,0]**3 + np.random.randn(len(x))*3  # 1 output
    return x, y

def load_MNIST():
    # Load MNIST dataset
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = mnist.target.astype(int)
    return X, y

import numpy as np

def create_sin_data_2D_output():
    """
    Generates synthetic data with 2 input features and 2 output targets.
    
    Returns:
      x: np.ndarray of shape (N, 2), where
         x[:,0] = linspace(-5,5,N)
         x[:,1] = 0.5 * x[:,0]
      y: np.ndarray of shape (N, 2), where
         y[:,0] = sin(x[:,0]) + noise
         y[:,1] = sin(x[:,1]) + noise
    """
    N = 200
    # build inputs
    x1 = np.linspace(-5, 5, N)
    x2 = 0.5 * x1
    x = np.column_stack((x1, x2))
    
    # build outputs with small Gaussian noise
    noise_std = 0.1
    y1 = np.sin(x1) + np.random.randn(N) * noise_std
    y2 = np.sin(x2) + np.random.randn(N) * noise_std
    y = np.column_stack((y1, y2))
    
    return x, y


def InitializeParameters(units):
    # Param initialization
    params = {"mw" : [], "Sw" : [], "mb" : [], "Sb" : []} # mean and variance for weights and biases

    for i in range(len(units)-1):
        # Initialize variance weights randomly with values from positive normal distribution
        mw_row, Sw_row, mb, Sb = [], [], [], []
        for j in range(units[i]):
            mw_col , Sw_col = [], []
            for k in range(units[i+1]):
                Sw_col.append(0.5 / units[i])
                mw_col.append(Sw_col[-1]**0.5 * np.random.randn(1))

            mw_row.append(mw_col)
            Sw_row.append(Sw_col)
  
        for j in range(units[i+1]):
            mb.append(0)
            Sb.append(0.5)

        params["mw"].append(mw_row)
        params["Sw"].append(Sw_row)
        params["mb"].append(mb)
        params["Sb"].append(Sb)
        

    '''
    # Print shape of parameters for each layer
    for i in range(len(units)-1):
        print("Layer", i)
        print("mw", i, params["mw"][i].shape)
        print("Sw", i, params["Sw"][i].shape)
        print("mb", i, params["mb"][i].shape)
        print("Sb", i, params["Sb"][i].shape)
    '''
        
    return params

def InitializeStates(units):
    # States initialization
    states = {"mz" : [], "Sz" : [], "ma" : [], "Sa" : [], "J" : [], "cov_yz" : [], "cov_z_w" : [], "cov_z_b" : [], "cov_z_z" : []}

    for i in range(len(units)):
        mz = []
        Sz = []
        ma = []
        Sa = []
        J = []
        cov_yz = []

        for j in range(units[i]):
            mz.append(0)
            Sz.append(0)
            ma.append(0)
            Sa.append(0)
            J.append(0)
            cov_yz.append(0)

        states["mz"].append(mz)
        states["Sz"].append(Sz)
        states["ma"].append(ma)
        states["Sa"].append(Sa)
        states["J"].append(J)
        states["cov_yz"].append(cov_yz)
        
    return states

def InitializeCovariances(units):
# Param initialization
    cov = {"cov_z_w" : [], "cov_z_z" : [], "cov_z_b" : []} # mean and variance for weights and biases

    for i in range(len(units)-1):
        # Initialize variance weights randomly with values from positive normal distribution
        cov_z_w_row, cov_z_z_row, cov_z_b = [], [], []
        for j in range(units[i]):
            cov_z_w_col , cov_z_z_col = [], []
            for k in range(units[i+1]):
                cov_z_z_col.append(0)
                cov_z_w_col.append(0)

            cov_z_w_row.append(cov_z_w_col)
            cov_z_z_row.append(cov_z_z_col)
  
        for j in range(units[i+1]):
            cov_z_b.append(0)

        cov["cov_z_w"].append(cov_z_w_row)
        cov["cov_z_z"].append(cov_z_z_row)
        cov["cov_z_b"].append(cov_z_b)

    return cov

def feed_forward(x, units, params, states, cov):
    for j in range(units[0]):
            states["ma"][0][j] = x[j]
            states["Sa"][0][j] = 0

    # For every layer
    for i in range(0, len(units)-1):
        no = units[i+1]
        ni = units[i]

        # For every unit in layer
        for j in range(no):
            sum_mz = 0
            sum_Sz = 0
            ma = 0
            Sa = 0
            # Calculate mean and variance of z
            # For every unit in previous layer    
            for k in range(ni):
                # mu_z = mu_w * mu_a
                ma = states["ma"][i][k]
                sum_mz += params["mw"][i][k][j] * ma

                Sa = states["Sa"][i][k]
                sum_Sz += (params["mw"][i][k][j] * params["mw"][i][k][j] * Sa) \
                                       + params["Sw"][i][k][j] * Sa \
                                       + (params["Sw"][i][k][j] * ma * ma)
            
            states["mz"][i+1][j] = sum_mz + params["mb"][i][j]
            states["Sz"][i+1][j] = sum_Sz + params["Sb"][i][j]

        # Pass through activation function
        # For every unit in layer
        if (i == len(units)-2):
            for j in range(no):
                states["ma"][i+1][j] = states["mz"][i+1][j]
                states["Sa"][i+1][j] = states["Sz"][i+1][j]
                states["J"][i+1][j] = 1
        else:
            for j in range(no):
                relu_aux = ReLU(states["mz"][i+1][j])
                states["ma"][i+1][j] = relu_aux

                if (relu_aux == 0):
                    states["Sa"][i+1][j] = 0
                    states["J"][i+1][j] = 0
                else:
                    states["Sa"][i+1][j] = states["Sz"][i+1][j]
                    states["J"][i+1][j] = 1

        for j in range(ni):
            for k in range(no):
                cov["cov_z_w"][i][j][k] = states["ma"][i][j] * params["Sw"][i][j][k]
                cov["cov_z_b"][i][k] = params["Sb"][i][k]
                cov["cov_z_z"][i][j][k] = states["Sz"][i][j] * states["J"][i][j] * params["mw"][i][j][k]

    
    # states["cov_yz"][-1][0] = states["Sz"][-1][0]
    for j in range(units[-1]):
        states["cov_yz"][-1][j] = states["Sz"][-1][j]

    return params, states, cov

def states_feed_backward(y, params, units, states, cov, sigma_v):
    # Init deltas
    deltas = {"dmz" : [], "dSz" : [], "dJ" : [], "dmw" : [], "dSw" : [], "dmb" : [], "dSb" : []}

    for i in range(len(units)):
        dmz = []
        dSz = []
        dJ = []

        for j in range(units[i]):
            dmz.append(0)
            dSz.append(0)
            dJ.append(0)

        deltas["dmz"].append(dmz)
        deltas["dSz"].append(dSz)
        deltas["dJ"].append(dJ)
    
    for i in range(len(units)-1):
        # Initialize variance weights randomly with values from positive normal distribution
        dmw_row, dSw_row, dmb, dSb = [], [], [], []
        for j in range(units[i]):
            dmw_col , dSw_col = [], []
            for k in range(units[i+1]):
                dSw_col.append(0)
                dmw_col.append(0)

            dmw_row.append(dmw_col)
            dSw_row.append(dSw_col)
  
        for j in range(units[i+1]):
            dmb.append(0)
            dSb.append(0)

        deltas["dmw"].append(dmw_row)
        deltas["dSw"].append(dSw_row)
        deltas["dmb"].append(dmb)
        deltas["dSb"].append(dSb)

    # my = states["mz"][-1][0]
    # Sy = states["Sz"][-1][0] + sigma_v**2

    # deltas["dmz"][-1][0] = states["mz"][-1][0] + (states["cov_yz"][-1][0] / Sy) * (y - my)
    # deltas["dSz"][-1][0] = states["Sz"][-1][0] - (states["cov_yz"][-1][0] / Sy) * states["cov_yz"][-1][0] 

    for j in range(units[-1]):
        my = states["mz"][-1][j]
        Sy = states["Sz"][-1][j] + sigma_v**2

        cov_yz = states["cov_yz"][-1][j]
        deltas["dmz"][-1][j] = my + (cov_yz / Sy) * (y[j] - my)
        deltas["dSz"][-1][j] = states["Sz"][-1][j] - (cov_yz / Sy) * cov_yz


    #############################
    # Feed backward the weights #
    #############################
    for i in range(len(units)-2, -1, -1):
        no = units[i+1]
        ni = units[i]
        for j in range(ni):
            for k in range(no):
                Jw = cov["cov_z_w"][i][j][k] / states["Sz"][i+1][k]
                deltas["dmw"][i][j][k] = params["mw"][i][j][k] + Jw * (deltas["dmz"][i+1][k] - states["mz"][i+1][k])
                deltas["dSw"][i][j][k] = params["Sw"][i][j][k] + Jw * (deltas["dSz"][i+1][k] - states["Sz"][i+1][k]) * Jw
        for j in range(no):
            Jb = cov["cov_z_b"][i][j] / states["Sz"][i+1][k]
            deltas["dmb"][i][j] = params["mb"][i][j] + Jb * (deltas["dmz"][i+1][j] - states["mz"][i+1][j])
            deltas["dSb"][i][j] = params["Sb"][i][j] + Jb * (deltas["dSz"][i+1][j] - states["Sz"][i+1][j]) * Jb

    ###################################
    # Feed backward the hidden states #
    ###################################
        if (i > 0):
            # For every unit in layer
            for j in range(ni):
                aux_mz = 0
                aux_Sz = 0
                for k in range(no):
                    Jz = cov["cov_z_z"][i][j][k] / states["Sz"][i+1][k]
                    aux_mz += Jz * (deltas["dmz"][i+1][k] - states["mz"][i+1][k])
                    aux_Sz += Jz * (deltas["dSz"][i+1][k] - states["Sz"][i+1][k]) * Jz

                deltas["dmz"][i][j] = states["mz"][i][j] + aux_mz
                deltas["dSz"][i][j] = states["Sz"][i][j] + aux_Sz
    
    # Update hidden states
    for i in range(len(units)-1):
        states["mz"][i+1] = deltas["dmz"][i+1]
        states["Sz"][i+1] = deltas["dSz"][i+1]
        params["mw"][i] = deltas["dmw"][i]
        params["Sw"][i] = deltas["dSw"][i]
        params["mb"][i] = deltas["dmb"][i]
        params["Sb"][i] = deltas["dSb"][i]
        
    return params, states, cov
    
import numpy as np
import matplotlib.pyplot as plt

def plot_2in2out_predictions(x_test, y_test, y_pred, sy_pred, sort_by=0):
    """
    x_test : array (N, 2)       test inputs
    y_test : array (N, 2)       true outputs
    y_pred : array (N, 2)       predicted means
    sy_pred: array (N, 2)       predicted std-devs
    sort_by: which input dim to sort/plot along (0 or 1)
    """
    # 1) sort so curves are smooth
    idx = np.argsort(x_test[:, sort_by])
    x = x_test[idx, sort_by]
    yt = y_test[idx]
    yp = y_pred[idx]
    s  = sy_pred[idx]

    # 2) make two stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    labels = ['Output 1', 'Output 2']
    colors = ['C0', 'C1']

    for m, ax in enumerate(axes):
        # true vs predicted
        ax.scatter(x, yt[:, m], color=colors[m], s=20, label='true')
        ax.plot(   x, yp[:, m], '-', color=colors[m], label='pred')
        # ±1σ band
        ax.fill_between(
            x,
            yp[:, m] - s[:, m],
            yp[:, m] + s[:, m],
            color=colors[m],
            alpha=0.3
        )
        ax.set_ylabel(labels[m])
        ax.legend(loc='best')

    axes[-1].set_xlabel(f'Input dimension {sort_by}')
    plt.tight_layout()
    plt.show()


def main():
    # x, y = create_sin_data()
    # x, y = create_sin_data_2D_input()
    # x, y = create_sin_data_2D_output()
    x, y = load_MNIST()
    
    print("x shape", x.shape)
    print("y shape", y.shape)

    # Divide data into training and test sets (80/20) randomly
    # x_train is obtained randomly from x
    # y_train is obtained from y according to the index of x_train
    idx = np.random.permutation(len(x))
    x_train = x[idx[:int(len(x)*0.8)]]
    y_train = y[idx[:int(len(y)*0.8)]]

    x_test = x[idx[int(len(x)*0.8):]]
    y_test = y[idx[int(len(y)*0.8):]]

    # Normalize data
    x_norm = Normal(np.mean(x_train), np.std(x_train))
    y_norm = Normal(np.mean(y_train), np.std(y_train))
    print("x_norm", x_norm.mean, x_norm.std)

    x_train = (x_train - x_norm.mean)/x_norm.std
    y_train = (y_train - y_norm.mean)/y_norm.std


    x_test = (x_test - x_norm.mean)/x_norm.std
    y_test = (y_test - y_norm.mean)/y_norm.std

    # Plot data
    #import matplotlib.pyplot as plt
    #plt.plot(x, y, 'o')
    #plt.show()
    
    #units = [1, 50, 1] # input, hidden, output
    units = [784, 50, 50, 10] # input, hidden, output

    # Training
    epochs = 10
    iterations = len(x_train)
    #iterations = 1
    sigma_v = 0.01

    # Initialize parameters

    params = InitializeParameters(units)
    states = InitializeStates(units)
    cov = InitializeCovariances(units)

    for i in range(epochs):

        for j in range(iterations):
            # Shuffle data
            idx = np.random.permutation(len(x_train))
            x_train = x_train[idx]
            y_train = y_train[idx]

            # Create batches
            x_batch = x_train[j]
            y_batch = y_train[j]

            # One hot encoding
            y_batch = np.zeros((len(y_batch), 10))
            for k in range(len(y_batch)):
                y_batch[k][y_train[j][k]] = 1


            # Feed forward
            params, states, cov = feed_forward(x_batch, units, params, states, cov)

            # Feed backward
            params, states, cov = states_feed_backward(y_batch, params, units, states, cov, sigma_v)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

    # 1. Compute predictions on the test set
    y_pred = np.zeros((len(x_test), 2))
    sy_pred = np.zeros((len(x_test), 2))
    for i in range(len(x_test)):
        feed_forward(x_test[i], units, params, states, cov)
        # grab both outputs
        for m in range(2):
            y_pred[i,m]  = states["mz"][-1][m]
            sy_pred[i,m] = np.sqrt(states["Sz"][-1][m])

    y_test = np.array(y_test)   # assume shape (N,2) as well

    # 2. Build a grid over the two input dims
    n_grid = 50
    x1_vals = np.linspace(x_test[:,0].min(), x_test[:,0].max(), n_grid)
    x2_vals = np.linspace(x_test[:,1].min(), x_test[:,1].max(), n_grid)
    X1, X2   = np.meshgrid(x1_vals, x2_vals)

    # 3. Evaluate the model on each grid point for both outputs
    Yg = np.zeros((n_grid, n_grid, 2))
    SYg = np.zeros((n_grid, n_grid, 2))
    for i in range(n_grid):
        for j in range(n_grid):
            x_pt = np.array([X1[i,j], X2[i,j]])
            feed_forward(x_pt, units, params, states, cov)
            for m in range(2):
                Yg[i,j,m]  = states["mz"][-1][m]
                SYg[i,j,m] = np.sqrt(states["Sz"][-1][m])

    # 4. Plot each output in its own 3D subplot
    fig = plt.figure(figsize=(14,6))
    for m in range(2):
        ax = fig.add_subplot(1, 2, m+1, projection='3d')
        surf = ax.plot_surface(
            X1, X2, Yg[:,:,m],
            alpha=0.7, cmap='viridis', edgecolor='none'
        )
        # scatter true test points for this output
        ax.scatter(
            x_test[:,0], x_test[:,1], y_test[:,m],
            color='red', s=30, label=f'Test pts (out {m+1})'
        )
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel(f'y₍{m+1}₎')
        ax.set_title(f'Output {m+1}: Predicted Mean')
        ax.legend()
        fig.colorbar(surf, ax=ax, shrink=0.5, label=f'ŷ₍{m+1}₎')

    plt.tight_layout()
    plt.show()

    # 5. (Optional) contour‐plot the stddev for each output
    fig2, axes = plt.subplots(1, 2, figsize=(14,5))
    for m, ax in enumerate(axes):
        c = ax.contourf(
            X1, X2, SYg[:,:,m],
            levels=20, cmap='magma'
        )
        ax.scatter(x_test[:,0], x_test[:,1], c='white', edgecolor='k')
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
        ax.set_title(f'Output {m+1}: σ_y')
        fig2.colorbar(c, ax=ax, label=f'σ₍{m+1}₎')
    plt.tight_layout()
    plt.show()

    # Testing    
    y_pred = []
    sy_pred = []
    for i in range(len(x_test)):
        # Feed forward
        feed_forward(x_test[i], units, params, states, cov)

        # Take mean of last layer
        y_pred.append(states["mz"][-1])
        sy_pred.append(np.sqrt(states["Sz"][-1]))

    # Unnormalize data
    #for i in range(len(y_pred)):
    #    y_pred[i] = y_pred[i]*y_norm.std + y_norm.mean

    print("y_pred shape", np.array(y_pred).shape)
    print("sy_pred shape", np.array(sy_pred).shape)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred).reshape(-1, units[-1])
    sy_pred = np.array(sy_pred).reshape(-1, units[-1])
    # Calculate MSE
    mse = np.mean((y_test - y_pred)**2)
    print("MSE", mse)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    sy_pred = np.array(sy_pred)

    # Sort the arrays based on x_test
    sorted_indices = np.argsort(x_test[:,0])
    x_test_sorted = x_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    sy_pred_sorted = sy_pred[sorted_indices]
    

    print("x_test_sorted shape", x_test_sorted.shape)
    print("y_test_sorted shape", y_test_sorted.shape)
    print("y_pred_sorted shape", y_pred_sorted.shape)
    print("sy_pred_sorted shape", sy_pred_sorted.shape)

    # Plot data 2D input
    import matplotlib.pyplot as plt
    # Plot data
    plt.plot(x_test_sorted[:,0], y_test_sorted[:,0], 'o')
    plt.plot(x_test_sorted[:,0], y_pred_sorted[:,0], 'o')
    plt.plot(x_test_sorted[:,1], y_test_sorted[:,1], 'o')
    plt.plot(x_test_sorted[:,1], y_pred_sorted[:,1], 'o')
    # Plot variances
    plt.fill_between(x_test_sorted[:,0], y_pred_sorted[:,0] - sy_pred_sorted[:,0].flatten(), y_pred_sorted[:,0] + sy_pred_sorted[:,0].flatten(), color='red', alpha=0.3)
    plt.fill_between(x_test_sorted[:,1], y_pred_sorted[:,1] - sy_pred_sorted[:,1].flatten(), y_pred_sorted[:,1] + sy_pred_sorted[:,1].flatten(), color='blue', alpha=0.3)

    # Add title and axis names
    plt.title('Sine function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['test', 'pred', 'test2', 'pred2'], loc='upper left')
    plt.show()

    y_test  = np.array(y_test)
    y_pred  = np.vstack(y_pred)   # shape (N,2)
    sy_pred = np.vstack(sy_pred)  # shape (N,2)

    plot_2in2out_predictions(x_test, y_test, y_pred, sy_pred, sort_by=0)


    
    

if __name__ == "__main__":
    main()
