import numpy as np

class Normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

def ReLU(x):
    return np.maximum(0, x)

def create_sin_data():
    x = np.linspace(-2*np.pi, 2*np.pi)
    y = np.sin(x) + np.random.randn(len(x))*0.2
    return x, y


def feed_forward(x, params, units):
    # Init states
    states = {"mz" : [], "Sz" : [], "ma" : [], "Sa" : [], "J" : []}

    for i in range(np.sum(units) * len(x)):
        states["mz"].append(x[i%len(x)])
        states["Sz"].append(0)
        states["ma"].append(x[i%len(x)])
        states["Sa"].append(0)
        states["J"].append(1)
     
    # For each layer
    for layer in range(len(units)-1):
        sum = 0
        ma_tmp = 0
        Sa_tmp = 0

        # For each output hidden unit
        for ou in range(units[layer+1]):

            # For each batch
            for batch in range(len(x)):
                sum = 0

                # For each input hidden unit
                for iu in range(units[layer]):
                    ma_tmp = states["ma"][batch * units[layer] + iu]
                    Sa_tmp = states["Sa"][batch * units[layer] + iu]
                    sum += params["mw"][layer][iu, ou] * params["mw"][layer][iu, ou]
                    + params["Sw"][layer][iu, ou] * Sa_tmp
                    + params["Sw"][layer][iu, ou] * ma_tmp * ma_tmp
                
                states["mz"][batch * units[layer+1] + ou] = sum + params["mb"][layer][ou]
                states["Sz"][batch * units[layer+1] + ou] = sum + params["Sb"][layer][ou]

        
        # Activate hidden units
        zeroPad = 0
        onePad = 1
        tmp = 0
        for j in range(len(x) * units[layer]):
            tmp = np.max(states["mz"][j], zeroPad)
            states["ma"][j] = tmp
            if tmp == 0:
                states["J"][j] = zeroPad
                states["Sa"][j] = zeroPad
            else:
                states["J"][j] = onePad
                states["Sa"][j] = states["Sz"][j]

    return states

def feed_backward(y, params, units, states):
    # Init deltas
    deltas = {"dmz" : [], "dSz" : [], "dma" : [], "dSa" : [], "dJ" : []}

    for i in range(np.sum(units) * len(y)):
        deltas["dmz"].append(0)
        deltas["dSz"].append(0)
        deltas["dma"].append(0)
        deltas["dSa"].append(0)
        deltas["dJ"].append(0)
    
'''
    float zeroPad = 0;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = (J[col + z_pos] * Sz[col + z_pos]) / (Sa[col + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[col] = zeroPad;
            delta_Sz[col] = zeroPad;
        } else {
            delta_mz[col] = tmp * (y[col] - ma[col + z_pos]);
            delta_Sz[col] = -tmp * (J[col + z_pos] * Sz[col + z_pos]);
        }
    }
'''
     


def main():
    x, y = create_sin_data()

    # Divide data into training and test sets
    x_train = x[:40]
    y_train = y[:40]
    x_test = x[40:]
    y_test = y[40:]

    # Normalize data
    x_norm = Normal(np.mean(x_train), np.std(x_train))
    y_norm = Normal(np.mean(y_train), np.std(y_train))

    x_train = (x_train - x_norm.mean)/x_norm.std
    y_train = (y_train - y_norm.mean)/y_norm.std

    x_test = (x_test - x_norm.mean)/x_norm.std
    y_test = (y_test - y_norm.mean)/y_norm.std

    
    units = [1, 50, 1] # input, hidden, output

    # Initialize parameters

    # Param initialization
    params = {"mw" : [], "Sw" : [], "mb" : [], "Sb" : []} # mean and variance for weights and biases

    for i in range(len(units)-1):
        params["mw"].append(np.random.randn(units[i], units[i+1]) * np.sqrt(1/units[i]))
        params["Sw"].append(np.random.randn(units[i], units[i+1]) * np.sqrt(1/units[i]))

        params["mb"].append(np.random.randn(units[i+1]) * np.sqrt(1/units[i]))
        params["Sb"].append(np.random.randn(units[i+1]) * np.sqrt(1/units[i]))



    # Training
    epochs = 1
    batch_size = 5
    iterations = int(len(x_train)/batch_size)

    for i in range(epochs):

        for j in range(1):
            # Shuffle data
            idx = np.random.permutation(len(x_train))
            x_train = x_train[idx]
            y_train = y_train[idx]

            # Create batches
            x_batch = x_train[j*batch_size:(j+1)*batch_size]
            y_batch = y_train[j*batch_size:(j+1)*batch_size]

            # Feed forward
            states = feed_forward(x_batch, params, units)

            # Feed backward
            feed_backward(y_batch, params, units, states)


if __name__ == "__main__":
    main()
