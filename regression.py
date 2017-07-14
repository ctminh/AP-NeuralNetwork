# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
np.random.seed(seed=1)
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
import math
from NN import *
rng = np.random.RandomState(1311)

def f_function(nb_of_samples):
    # Target function
    x = np.arange(0, 2 * math.pi, 0.001)
    t = np.sin(x)

    # Create the dataset with gaussian noise based on target function
    x1 = np.random.uniform(0, 1, nb_of_samples)
    noise_variance = 0.2
    noise = np.random.randn(x1.shape[0]) * noise_variance
    y = np.sin(2 * np.pi * 1 * x1 + 0) + noise
    dataX = 2 * np.pi * 1 * x1 + 0
    return (dataX, y)

def devideDataset(data1, label1, numOfTrain, numOfValidation):
    train_X = data1[0:numOfTrain]

    train_Y = label1[0:numOfTrain]

    val_X = data1[numOfTrain:(numOfTrain + numOfValidation)]

    val_Y = label1[numOfTrain:(numOfTrain + numOfValidation)]

    test_X = data1[(numOfTrain + numOfValidation):]

    test_Y = label1[(numOfTrain + numOfValidation):]

    return (train_X, train_Y, val_X, val_Y, test_X, test_Y)

def train_draw(config, all_cost, i, cost, x_visualize, y_visualize, y_true):
    # Display the training process
    lr = config['lr']
    num_hidden_node = config['num_hidden_node']

    pylab.clf()
    f = plt.figure(2, figsize=(16, 8))

    title = 'Regression with %d hidden nodes, lr = %.4g, %d iter, cost = %.4g' % (num_hidden_node, lr, i, cost)

    plt.subplot(1, 2, 1)
    plt.plot(x_visualize, y_visualize, 'b')
    plt.plot(x_visualize, y_true, 'r')
    # [grid1, grid2, grid3] = find_decision_boundary(X, Y, W1, b1, W2, b2, config)

    # visualize_decision_grid(grid1, grid2, grid3, 2)

    # visualize_data(X[0:num_train_per_class, :],
    #                X[num_train_per_class:num_train_per_class * 2, :],
    #                X[num_train_per_class * 2:, :],2)
    plt.subplot(1, 2, 2)
    plt.plot(all_cost, 'b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')

    f.suptitle(title, fontsize=15)
    plt.pause(0.1)

    pylab.draw()

if __name__ == '__main__':
    # Create dataset
    (X, Y) = f_function(1000)

    # For visualize:
    x_visualize = np.arange(0, 2 * math.pi, 0.005)
    y_true = np.sin(x_visualize)

    # Draw data
    # plt.plot(X, Y, 'bo')
    # plt.axis([0, 7, -2, 2])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # Devide dataset
    numOfTrain = 500
    numOfValidation = 250
    train_X, train_Y, val_X, val_Y, test_X, test_Y = devideDataset(X, Y, numOfTrain, numOfValidation)
    # print len(train_X)
    # print len(val_X)
    # print len(test_X)

    # Define a sample model to be trained on the data
    hidden_neurons_1 = 24  # Number of neurons in the first hidden-layer
    hidden_neurons_2 = 24
    # Create the model
    layers = []
    # Add first hidden layer
    layers.append(LinearLayer(1, hidden_neurons_1))
    layers.append(NonLinearLayer())
    # Add second hidden layer
    layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2))
    layers.append(NonLinearLayer())
    # # Add output layer
    layers.append(LinearLayer(hidden_neurons_2, 1))
    layers.append(OutputLayer())

    # print "layer 0 - W1", layers[0].W
    # print "layer 0 - b1", layers[0].b
    # print "layer 2 - W2", layers[2].W
    # print "layer 2 - b2", layers[2].b

    gradient_checking(layers, train_X, train_Y, True)

    # Create mini-batches

    # initalize some lists to store the cost for future analysis        
    all_costs = []

    max_nb_of_iterations = 5000  # Train for a maximum of 300 iterations
    learning_rate = 0.001  # Gradient descent learning rate
    X = train_X.reshape(train_X.shape[0],1)
    Y = train_Y.reshape(train_Y.shape[0],1)
    config = {}
    config['num_hidden_node'] = hidden_neurons_1
    config['lr'] = learning_rate

    for iteration in range(max_nb_of_iterations):
        activations = forward_step(X, layers)

        cost = layers[-1].get_cost(activations[-1], Y)  # Get cost
        all_costs.append(cost)

        param_grads = backward_step(activations, Y, layers)  # Get the gradients
        update_params(layers, param_grads, learning_rate)

        print('cost at iteration %d is %f' %(iteration, cost))
        if(iteration % 20 == 0):
            activations = forward_step(x_visualize.reshape(x_visualize.shape[0],1), layers)
            y_visualize = activations[-1]
            train_draw(config, all_costs, iteration, cost, x_visualize, y_visualize, y_true)


# forward step
# activations = forward_step(x_visualize.reshape(x_visualize.shape[0],1), layers)
# y_visualize = activations[-1]


# Draw data

# plt.plot(x_visualize, y_visualize, 'b')
# plt.plot(x_visualize, y_true, 'r')

# plt.axis([0, 7, -2, 2])
# plt.xlabel('x')
# plt.ylabel('y')
plt.show()

