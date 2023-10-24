import numpy as np
from tagi import network


def create_sin_data():
    x = np.linspace(-2*np.pi, 2*np.pi)
    y = np.sin(x) + np.random.randn(len(x))*0.2
    return x, y


def main():
    x, y = create_sin_data()

    # Divide data into training and test sets
    x_train = x[:40]
    y_train = y[:40]
    x_test = x[40:]
    y_test = y[40:]

    # Neural network with one hidden layer of 50 units

    mw1 = np.random.randn(1, 50) * 0.1  # Mean weights for hidden layer
    Sw1 = np.random.randn(1, 50) * 0.1  # Standard dev. weights of hidden layer

    mb1 = np.zeros((1, 50))  # Mean biases for hidden layer
    Sb1 = np.zeros((1, 50))  # Standard deviation biases for hidden layer

    mw2 = np.random.randn(50, 1) * 0.1  # Mean weights for output layer
    Sw2 = np.random.randn(50, 1) * 0.1  # Standard dev. weights of output layer

    mb2 = np.zeros((1, 1))  # Mean biases for output layer
    Sb2 = np.zeros((1, 1))  # Standard deviation biases for output layer

    # Training
    epochs = 50
    batch_size = 10
    iterations = int(len(x_train)/batch_size)

    for i in range(epochs):

        for j in range(iterations):
            # Shuffle data
            idx = np.random.permutation(len(x_train))
            x_train = x_train[idx]
            y_train = y_train[idx]

            # Create batches
            x_batch = x_train[j*batch_size:(j+1)*batch_size]
            y_batch = y_train[j*batch_size:(j+1)*batch_size]

            # Feed forward


if __name__ == "__main__":
    main()
