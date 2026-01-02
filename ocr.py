import numpy as np
import math
import json
import os

class OCRNeuralNetwork:
    NN_FILE_PATH = 'nn.json'
    LEARNING_RATE = 0.01

    def __init__(self, num_hidden_nodes, use_file=True):
        self.num_hidden_nodes = num_hidden_nodes
        self._use_file = use_file
        
        # Initialize weights with correct shapes (Rows, Cols)
        self.theta1 = self._rand_initialize_weights(num_hidden_nodes, 400)
        self.theta2 = self._rand_initialize_weights(10, num_hidden_nodes)
        self.input_layer_bias = self._rand_initialize_weights(num_hidden_nodes, 1)
        self.hidden_layer_bias = self._rand_initialize_weights(10, 1)
        
        if self._use_file and os.path.isfile(self.NN_FILE_PATH):
            self._load()

    def _rand_initialize_weights(self, size_out, size_in):
        # Initialize strictly as matrices to avoid list/array confusion
        return np.asmatrix(np.random.uniform(-0.06, 0.06, (size_out, size_in)))

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def sigmoid(self, z):
        return np.vectorize(self._sigmoid_scalar)(z)

    def sigmoid_prime(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def train(self, training_data_array):
        for data in training_data_array:
            # Forward Propagation
            # Input data assumed to be list, convert to (400, 1) matrix
            y0 = np.asmatrix(data['y0']).T
            
            y1 = np.dot(self.theta1, y0)
            y1 = y1 + self.input_layer_bias # Shape: (15, 1)
            y1 = self.sigmoid(y1)

            y2 = np.dot(self.theta2, y1)
            y2 = y2 + self.hidden_layer_bias # Shape: (10, 1)
            y2 = self.sigmoid(y2)

            # Back Propagation
            actual_vals = [0] * 10
            actual_vals[data['label']] = 1
            
            output_errors = np.asmatrix(actual_vals).T - y2
            
            hidden_errors = np.multiply(np.dot(self.theta2.T, output_errors), self.sigmoid_prime(y1))

            # Update Weights
            self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, y0.T)
            self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.T)
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors

    def predict(self, test):
        y0 = np.asmatrix(test).T 
        
        y1 = np.dot(self.theta1, y0)
        y1 = y1 + self.input_layer_bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(self.theta2, y1)
        y2 = y2 + self.hidden_layer_bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        index = results.index(max(results))
        confidence = float(max(results))

        return index, confidence

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        with open(self.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(self.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        # Convert lists back to matrices
        self.theta1 = np.asmatrix(nn['theta1'])
        self.theta2 = np.asmatrix(nn['theta2'])
        self.input_layer_bias = np.asmatrix(nn['b1'])
        self.hidden_layer_bias = np.asmatrix(nn['b2'])