import numpy as np
import matplotlib.pyplot as plt

#data
data_train_in = np.loadtxt("data/mnist_small_train_in.txt", delimiter=',').T
data_train_out = np.loadtxt("data/mnist_small_train_out.txt", delimiter=',')
data_test_in = np.loadtxt("data/mnist_small_test_in.txt", delimiter=',').T
data_test_out = np.loadtxt("data/mnist_small_test_out.txt", delimiter=',')


#hot encode
data_train_out = np.eye(10)[data_train_out.astype('int32')].T
data_test_out = np.eye(10)[data_test_out.astype('int32')].T

#meta params
NUMBER_INPUT_NEURONS = 784
NUMBER_HIDDEN_LAYER_NEURONS = 250
NUMBER_OUTPUT_NEURONS = 10
SIZE_OF_MINI_BATCH = 30
NUM_OF_BATCHES = -(-data_train_in.shape[1] // SIZE_OF_MINI_BATCH)
STEP_SIZE = 5
EPOCHS = 15

'----------- Helper functions -----------'


def initialize():
    'initialize weights and biases'
    w_0 = np.random.standard_normal((NUMBER_HIDDEN_LAYER_NEURONS, NUMBER_INPUT_NEURONS))
    w_1 = np.random.standard_normal((NUMBER_OUTPUT_NEURONS, NUMBER_HIDDEN_LAYER_NEURONS))
    b_0 = np.random.standard_normal((NUMBER_HIDDEN_LAYER_NEURONS, 1))
    b_1 = np.random.standard_normal((NUMBER_OUTPUT_NEURONS, 1))
    return w_0, w_1, b_0, b_1


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def cross_entropy(y, y_hat):
    m = y.shape[1]
    likelihood_sum = np.sum(np.multiply(y, np.log(y_hat)))
    c_e = -(1 / m) * likelihood_sum
    return c_e


def shuffle_data():
    shuffled_arr = np.random.permutation(data_train_in.shape[1]).astype(int)
    data_in_new = data_train_in[:, shuffled_arr]
    data_out_new = data_train_out[:, shuffled_arr]
    return data_in_new, data_out_new


def evaluate_prediction(y, y_hat):
    y = y.T
    y_hat = y_hat.T
    correct_prediction = 0
    for i in range(y.shape[0]):
        prediction = np.argmax(y_hat[i])
        if y[i][prediction] == 1.0:
            correct_prediction += 1
    return correct_prediction/y.shape[0]


'------- Propagation -------------'


def forward_propagation(w_0, w_1, b_0, b_1, x):
    'forwardpropagation'
    z_0 = np.matmul(w_0, x) + b_0
    a_0 = sigmoid(z_0)
    z_1 = np.matmul(w_1, a_0) + b_1
    y_hat = np.exp(z_1) / np.sum(np.exp(z_1), axis=0)
    return z_0, z_1, a_0, y_hat


def backward_propagation(w_1, z_0, z_1, a_0, y_hat, x, y, size_of_batch):
    error_z_1 = y_hat - y
    der_w_1 = (1 / size_of_batch) * np.matmul(error_z_1, a_0.T)
    der_b_1 = (1 / size_of_batch) * np.sum(error_z_1, axis=1, keepdims=True)

    error_z_0 = np.matmul(w_1.T, error_z_1)
    der_z_0 = error_z_0 * sigmoid(z_0) * (1 - sigmoid(z_0))
    der_w_0 = (1 / size_of_batch) * np.matmul(der_z_0, x.T)
    der_b_0 = (1 / size_of_batch) * np.sum(der_z_0, axis=1, keepdims=True)
    return der_b_1, der_w_1, der_b_0, der_w_0


'----------------- Neural Network -------------------'


def neural_net():
    evolution = np.zeros(EPOCHS + 2)
    #randomly initialize weights, biases
    w_0, w_1, b_0, b_1 = initialize()
    a_0, a_1, z_0, y_hat = forward_propagation(w_0, w_1, b_0, b_1, data_test_in)
    evolution[0] = (1 - evaluate_prediction(data_test_out, y_hat))
    print("Misclassification rate in Epoch {}: {}:".format\
              (0, (1 - evaluate_prediction(data_test_out, y_hat))))
    epoch_arr = [1, 3, 8, 15, 20, 25]
    for i in range(1, EPOCHS):
        #shuffle data
        data_train_in_shuffled, data_train_out_shuffled = shuffle_data()
        for j in range(1, NUM_OF_BATCHES):
            'get the mini batch'
            start = j * SIZE_OF_MINI_BATCH
            end = min(start + SIZE_OF_MINI_BATCH, data_train_in.shape[1] - 1)
            x = data_train_in_shuffled[:, start:end]
            y = data_train_out_shuffled[:, start:end]
            size_mini_batch = end - start

            z_0, z_1, a_0, y_hat = forward_propagation(w_0, w_1, b_0, b_1, x)
            der_b_1, der_w_1, der_b_0, der_w_0 = backward_propagation\
                (w_1, z_0, z_1, a_0, y_hat, x, y, size_mini_batch)
            'learn'
            w_0 -= STEP_SIZE*der_w_0
            w_1 -= STEP_SIZE*der_w_1
            b_0 -= STEP_SIZE*der_b_0
            b_1 -= STEP_SIZE*der_b_1

            a_0, a_1, z_0, y_hat = forward_propagation(w_0, w_1, b_0, b_1, data_test_in)
            evolution[i] = (1 - evaluate_prediction(data_test_out, y_hat))
        'evaluate on test data'
        if i in epoch_arr:
            print("Misclassification rate in Epoch {}: {}:".format\
                      (i, (1 - evaluate_prediction(data_test_out, y_hat))))

    'evaluate on test data'
    a_0, a_1, z_0, y_hat = forward_propagation(w_0, w_1, b_0, b_1, data_test_in)
    print("Misclassification rate in Epoch {}: {}:".format\
              (EPOCHS, (1 - evaluate_prediction(data_test_out, y_hat))))
    print("\r\n")
    print("Cross Entropy value: {}".format(cross_entropy(data_test_out, y_hat)))
    plt.plot(range(EPOCHS + 2), evolution, label='Misclassification rate')
    plt.xlabel('Epochs')
    plt.ylabel('Misclassification rate')
    plt.legend()
    plt.title('Misclassification rate over the Epochs')
    plt.savefig("neural_network.png")
    plt.close()
    return w_0, w_1, b_0, b_1


data_train_in, data_train_out = shuffle_data()
data_train_in = data_train_in.T[:6000].T
data_test_out = data_test_out.T[:6000].T


c_w_0, c_w_1, c_b_0, c_b_1 = neural_net()


