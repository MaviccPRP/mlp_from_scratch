# Class of a NeuralNetwork with SGD and Momentum
import numpy as np
import matplotlib.pyplot as plt  # Plotting library
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from algebra_helpers import dot, hadamard, substract, scalar_mult, transpose, summarize, sum_matrix

class NeuralNetwork:
    def __init__(self, samples, labels, size_hidden=30, eta=0.1, my=0.9, epochs=10, optimizer="", verbose=False):
        '''

        :param samples: input samples
        :param labels: input labels
        :param size_hidden: number of units in hidden layer
        :param eta: learning rate
        :param my: learning factor for momentum
        :param epochs: number of epochs
        :param optimizer: type of optimizer ("sgd", "momentum", "adagrad")
        :param verbose: print accuracy and error per epoch
        '''
        self.samples = samples
        self.labels = labels
        self.w01 = np.random.random((len(self.samples[0]), size_hidden))
        self.w12 = np.random.random((size_hidden, len(self.labels[0])))
        self.v01 = np.zeros((len(self.samples[0]), size_hidden))
        self.v12 = np.zeros((size_hidden, len(self.labels[0])))
        self.g01 = np.zeros((len(self.samples[0]), size_hidden))
        self.g12 = np.zeros((size_hidden, len(self.labels[0])))
        self.b1 = np.array([0])
        self.b2 = np.array([0])
        self.eta = eta
        self.epochs = epochs
        self.my = my
        self.optimizer = optimizer
        self.verbose = verbose

    def sigmoid(self, x, deriv=False):
        if (deriv == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, deriv=False):
        if deriv == True:
            # Return the partial derivation of the activation function
            return np.multiply(x, 1 - x)
        y = x - np.max(x)
        e_x = np.exp(y)
        return e_x / e_x.sum()


    def relu(self, x, deriv=False):
        if deriv == True:
            return 1. * (x > 0)
        return x * (x > 0)

    def fit(self):
        '''
        Method to fit the input data and optimize the weights in the neural network
        :return:
        '''
        accuracy = []
        no_epochs = []

        if self.optimizer == "adagrad":
            # initialize matrix for adagrad
            gti_01 = np.zeros(len(self.w01[0]))
            gti_12 = np.zeros(len(self.w12[0]))

        for epoch in range(self.epochs):
            for i in range(0, len(self.samples), 1):
                l0 = self.samples[i:i + 1]
                y = self.labels[i:i + 1]

                # Feed Forward Pass
                l1 = self.relu(dot(l0, self.w01) + 1 * self.b1)
                l2 = self.softmax(dot(l1, self.w12) + 1 * self.b2)

                err = substract(y, l2)
                l2_error = (scalar_mult(0.5,hadamard(err,err)))

                l2_error_total = str(np.mean(np.abs(l2_error)))


                #if l2_error_total == 1.0:
                #    if self.verbose: print("Overflow")
                #    return

                # Backpropagation
                # dE_total/douto
                l2_delta = scalar_mult(-1, err)
                # douto/dneto = deriv activation
                l2_delta = hadamard(l2_delta, self.softmax(l2, deriv=True))
                # dneth/dw
                l2_delta = dot(transpose(l2_delta), l1)

                # dEo/neto
                # dEo/douto * douto/dneto
                l1_delta = sum_matrix(hadamard(scalar_mult(-1, err), self.softmax(l2, deriv=True)), axis=0)

                # dEo/outh
                # dEo/neto * dneto/douth
                l1_delta = hadamard(l1_delta, self.w12)

                # dEtotal/outh = Sum(Eo/outh)
                l1_delta = sum_matrix(l1_delta, axis=1)

                # douth/neth
                l1_delta = hadamard(l1_delta, self.relu(l1, deriv=True))

                # dneth/dw
                l1_delta = dot(transpose(l1_delta), l0)

                if self.optimizer == "adagrad":
                    # Update Weights using AdaGrad
                    grad_12 = l2_delta.T
                    self.g12 += np.power(grad_12, 2)
                    adjusted_grad = grad_12 / np.sqrt(0.0000001 + self.g12)
                    self.w12 = self.w12 - adjusted_grad

                    grad_01 = l1_delta.T
                    self.g01 += np.power(grad_01, 2)
                    adjusted_grad = grad_01 / np.sqrt(0.0000001 + self.g01)
                    self.w01 = self.w01 - adjusted_grad

                if self.optimizer == "sgd":
                    # Update Weights
                    self.w01 = substract(self.w01, scalar_mult(self.eta/((epoch+1)/50), transpose(l1_delta)))
                    self.w12 = substract(self.w12, scalar_mult(self.eta/((epoch+1)/50), transpose(l2_delta)))

                if self.optimizer == "momentum":
                    # Update Weights using Momentum
                    self.v01 = self.my * self.v01 + self.eta * l1_delta.T
                    self.w01 -= self.v01
                    self.v12 = self.my * self.v12 + self.eta * l2_delta.T
                    self.w12 -= self.v12

            if epoch % 1 == 0:
                if self.verbose:
                    y_pred, y_true = self.predict(X_test, y_test)
                    print("Epoch: ", epoch, " - Error: ", l2_error_total, " - Accuracy im Testset: ", accuracy_score(y_true, y_pred))
                    y_pred, y_true = self.predict(X_train, y_train)
                    #print("Epoch: ", epoch, " - Error: ", l2_error_total, " - Accuracy im Trainingsset: ", accuracy_score(y_true, y_pred))
                    #print("############################################")

                    y_pred, y_true = self.predict(X_test, y_test)
                    acc = accuracy_score(y_true, y_pred)
                    accuracy.append(acc)
                    no_epochs.append(epoch)

        if self.verbose:
            return no_epochs, accuracy

    def predict(self, test_samples, test_labels):
        '''
        Predict test data using the fitted model
        :param test_samples:
        :param test_labels:
        :return:
        '''
        l1 = self.relu(np.dot(test_samples, self.w01) + 1 * self.b1)
        l2 = self.softmax(np.dot(l1, self.w12) + 1 * self.b2)
        y_pred = (l2 == l2.max(axis=1)[:, None]).astype(float)
        res_pred = []
        res_labels = []

        def checkEqual1(iterator):
            iterator = iter(iterator)
            try:
                first = next(iterator)
            except StopIteration:
                return True
            return all(first == rest for rest in iterator)

        for k in y_pred:
            for i, j in enumerate(k):
                if int(j) == 1 and not checkEqual1(k):
                    res_pred.append(i)
                    break
                if checkEqual1(k):
                    res_pred.append(0)
                    break
        for k in test_labels:
            for i, j in enumerate(k):
                if j == 1.0:
                    res_labels.append(i)

        return res_pred, res_labels


# Prepare dataset and split into test and training data

# MNIST
mnist = fetch_mldata('MNIST original', data_home="./data")
samples = mnist.data
samples = samples/(len(samples)*10)
y = mnist.target.reshape((len(samples), 1))

enc = OneHotEncoder()
enc.fit(y)
labels = enc.transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.33, random_state=42)

# Create instance of NeuralNetwork, fit to dataset, predict and print accuracy

etas = [3.5]

for eta in etas:
    NN = NeuralNetwork(samples=X_train, labels=y_train, eta=eta, epochs=100, size_hidden=5, optimizer="sgd", verbose=True)
    NN.fit()
