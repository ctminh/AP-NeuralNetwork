# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
np.random.seed(seed=1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
from NN import *

def train_draw(iteration, minibatch_costs, training_costs, validation_costs, nb_of_batches, lr):
	# Display the training process
	pylab.clf()
	# f = plt.figure(2, figsize=(16, 8))
	f = plt.figure(1, figsize=(12, 6))

	# Plot the minibatch, full training set, and validation costs
	minibatch_x_inds = np.linspace(0, iteration, num=iteration * nb_of_batches)
	iteration_x_inds = np.linspace(1, iteration, num=iteration)
	# Plot the cost over the iterations
	# plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
	plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
	plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
	# Add labels to the plot
	plt.xlabel('iteration')
	plt.ylabel('$\\xi$', fontsize=15)
	plt.title('Decrease of cost over backprop iteration')
	plt.legend()
	x1,x2,y1,y2 = plt.axis()
	plt.axis((0,nb_of_iterations,0,2.5))
	plt.grid()

	title = 'Neural Netwoek with %d hidden nodes, lr = %.4g' % (20, lr)

	f.suptitle(title, fontsize=15)
	plt.pause(0.1)

	pylab.draw()

# load the data
digits = datasets.load_digits()

# Load the targets.
# Note that the targets are stored as digits, these need to be 
#  converted to one-hot-encoding for the output sofmax layer.
T = np.zeros((digits.target.shape[0],10))
T[np.arange(len(T)), digits.target] += 1

# Divide the data into a train and test set.
X_train, X_test, T_train, T_test = cross_validation.train_test_split(
    digits.data, T, test_size=0.3)

# Divide the test set into a validation set and final test set.
X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
    X_test, T_test, test_size=0.5)

# Plot an example of each image.
# fig = plt.figure(figsize=(10, 1), dpi=100)
# for i in range(10):
#     ax = fig.add_subplot(1,10,i+1)
#     ax.matshow(digits.images[i], cmap='binary') 
#     ax.axis('off')
# plt.show()

# Define a sample model to be trained on the data
hidden_neurons_1 = 20  # Number of neurons in the first hidden-layer
hidden_neurons_2 = 20  # Number of neurons in the second hidden-layer
# Create the model
layers = []
# Add first hidden layer
layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1))
layers.append(NonLinearLayer())
# Add second hidden layer
# layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2))
# layers.append(NonLinearLayer())
# Add output layer
layers.append(LinearLayer(hidden_neurons_1, T_train.shape[1]))
layers.append(OutputLayer())

# Perform gradient checking
gradient_checking(layers, X_train, T_train)


# Create mini-batches
# Create the minibatches
batch_size = 25  # Approximately 25 samples per batch
nb_of_batches = X_train.shape[0] / batch_size  # Number of batches
# Create batches (X,Y) from the training set
XT_batches = zip(
    np.array_split(X_train, nb_of_batches, axis=0),  # X samples
    np.array_split(T_train, nb_of_batches, axis=0))  # Y targets

# Perform backpropagation
# initalize some lists to store the cost for future analysis        
minibatch_costs = []
training_costs = []
validation_costs = []

max_nb_of_iterations = 300  # Train for a maximum of 300 iterations
learning_rate = 0.5  # Gradient descent learning rate

# Train for the maximum number of iterations
for iteration in range(max_nb_of_iterations):
    for X, T in XT_batches:  # For each minibatch sub-iteration
        activations = forward_step(X, layers)  # Get the activations
        minibatch_cost = layers[-1].get_cost(activations[-1], T)  # Get cost
        minibatch_costs.append(minibatch_cost)
        param_grads = backward_step(activations, T, layers)  # Get the gradients
        update_params(layers, param_grads, learning_rate)  # Update the parameters

    # Get full training cost for future analysis (plots)
    activations = forward_step(X_train, layers)
    train_cost = layers[-1].get_cost(activations[-1], T_train)
    training_costs.append(train_cost)
    # Get full validation cost
    activations = forward_step(X_validation, layers)
    validation_cost = layers[-1].get_cost(activations[-1], T_validation)
    validation_costs.append(validation_cost)
    # if len(validation_costs) > 3:
    #     # Stop training if the cost on the validation set doesn't decrease
    #     #  for 3 iterations
    #     if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
    #         break
    # print "[Training] iteration %d" %iteration
    # print "minibatch_costs = " ,minibatch_costs
    # print "training_costs = " ,training_costs
    # print "validation_costs = " ,validation_costs


    
nb_of_iterations = iteration + 1  # The number of iterations that have been executed

# # Plot the minibatch, full training set, and validation costs
# minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations*nb_of_batches)
# iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
# # Plot the cost over the iterations
# plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
# plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
# plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
# # Add labels to the plot
# plt.xlabel('iteration')
# plt.ylabel('$\\xi$', fontsize=15)
# plt.title('Decrease of cost over backprop iteration')
# plt.legend()
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,nb_of_iterations,0,2.5))
# plt.grid()
# plt.show()

# Get results of test data
y_true = np.argmax(T_test, axis=1)  # Get the target outputs
activations = forward_step(X_test, layers)  # Get activation of test samples
y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
print('The accuracy on the test set is {:.2f}'.format(test_accuracy))

print len(X_test)
print digits.images[100]
print y_true
print y_pred

# Test NN model with real image
fig = plt.figure(figsize=(10, 2), dpi=100)
ax = fig.add_subplot(1,10,1)
ax.matshow(digits.images[8], cmap='binary')
ax.axis('off')


plt.show()
