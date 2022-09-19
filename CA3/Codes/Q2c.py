import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(644)
xx = np.random.uniform(0, 2 * np.pi, 10000)
yy = np.random.uniform(0, 2 * np.pi, 10000)

x = (xx - min(xx)) / (max(xx) - min(xx))
y = (yy - min(yy)) / (max(yy) - min(yy))
label = np.sin(xx + yy)
df = pd.DataFrame([x, y]).transpose()
train = df.sample(frac=0.8, random_state=200)  # random state is a seed value
test = df.drop(train.index)
validation = train.sample(frac=0.2, random_state=200)
label_validation = label[validation.index]
ttrain = train.drop(validation.index)
label_train = label[ttrain.index]
label_test = label[test.index]


class MyNeuralNet():
    def __init__(self):
        # defining weights and error
        np.random.seed(200)
        self.weights = np.random.rand(2, 10)
        self.weights2 = np.random.rand(10, 1)
        self.b1 = np.random.rand(10, 1)
        self.b2 = 1
        self.erriter = np.zeros(5000)
        self.errtest = np.zeros(5000)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return (1 / np.cosh(x) ** 2)

    def train(self, train, training_out, test, test_out, training_iterations=5000):
        # training the model and recording errors in each iteration
        for i in range(training_iterations):
            for iteration in np.random.randint(len(train), size=100):
                training_data = train[iteration]
                training_labels = training_out[iteration]
                outputs = self.predict(training_data)
                output = outputs[1]
                output1 = outputs[0]
                error = training_labels - output
                weight2_update = np.dot(output1.T, error)
                weight1_update = np.dot(training_data.reshape(2, 1),
                                      error * self.tanh_derivative(np.dot(training_data,
                                                                          self.weights) + self.b1.T) * self.weights2.T)

                b2_update = error
                b1_update = error * self.tanh_derivative(np.dot(training_data, self.weights) + self.b1.T
                                                              ) * self.weights2.T

                self.weights2 = self.weights2 + 0.01 * weight2_update
                self.weights = self.weights + 0.01 * weight1_update
                self.b1 = self.b1 + 0.01 * b1_update.T
                self.b2 = self.b2 + 0.01 * b2_update
            # calculate MSE for train and test data and save them in array
            predicted_train = neural_network.predict(train)[1]
            self.erriter[i] = ((predicted_train.T - training_out) ** 2).mean()
            predicted = neural_network.predict(test)[1]
            self.errtest[i] = ((predicted.T - test_out) ** 2).mean()

    def predict(self, inputs):
        # passing the inputs via the neuron to get output
        inputs = inputs/1.0
        output1 = self.tanh(np.dot(inputs, self.weights) + self.b1.T)
        output = np.dot(output1, self.weights2) + self.b2
        return [output1, output]


neural_network = MyNeuralNet()
training_data = np.array(ttrain)
training_labels = np.array(label_train)
neural_network.train(training_data, training_labels, test, label_test, 5000)
plt.plot(neural_network.errtest)
plt.xlabel('iter', fontsize=18)
plt.ylabel('error', fontsize=16)
plt.show()
plt.plot(neural_network.erriter)
plt.xlabel('iter', fontsize=18)
plt.ylabel('error', fontsize=16)
plt.show()
val_pred = neural_network.predict(validation)[1]
val_error = ((val_pred.T - label_validation) ** 2).mean()
print(val_error, "MSE for validation data")
print(neural_network.erriter[4999], "MSE for train data")
print(neural_network.errtest[4999], "MSE for test data")

predsin = []
xtest = np.linspace(0, 2 * np.pi, 2000)
xtest_norm = (xtest - min(xtest)) / (max(xtest) - min(xtest))
ytest = np.zeros(2000)
data = np.array([xtest_norm, ytest])
predsin = neural_network.predict(data.T)[1]
predsin = np.array(predsin)
plt.plot(xtest, predsin, label="Predicted")
plt.plot(xtest, np.sin(xtest), label="Actual")
plt.legend(loc="upper right")
plt.xlabel('x', fontsize=18)
plt.show()
