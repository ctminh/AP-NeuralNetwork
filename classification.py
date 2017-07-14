# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pylab
import time
import matplotlib.text as plttext
from NN import *
rng = np.random.RandomState(1311)
# %matplotlib inline

# Generate dataset
def create_data(numOfSamples=None):
	"""
	Generate samples of 3 classes using normal distribution
	+ type(numOfSamples) int
	+ param numOfSamples: number of sample to be generated for each class
	+ return: data and label for each class
	"""
	I = np.eye(3, dtype=np.float32)
	if(numOfSamples == None):
	    numOfSamples = 100
	    
	# Generate first class
	m1 = np.asarray([0, 0], dtype=np.float32)
	cov1 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
	data1 = rng.multivariate_normal(m1, cov1, numOfSamples)
	label1 = np.ones((numOfSamples), dtype=np.uint16) - 1
	label1 = I[label1,:]

	# Generate second class
	m2 = np.asarray([5,5], dtype=np.float32)
	cov2 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
	data2 = rng.multivariate_normal(m2, cov2, numOfSamples)
	label2 = np.ones((numOfSamples), dtype=np.uint16)
	label2 = I[label2, :]

	# Generate third class
	noise = np.abs((np.reshape(rng.normal(0, 0.01, numOfSamples), (numOfSamples,1))))
	S1 = np.asarray([[1, 0], [0, 0.7]], dtype=np.float32)
	S2 = np.asarray([[4, 0], [0, 4]], dtype=np.float32)
	m3 = np.asarray([0.5, 0.5], dtype=np.float32)
	cov3 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
	data3 = rng.multivariate_normal(m3, cov3, numOfSamples)
	data3 = data3/np.repeat(np.sqrt(np.sum(data3**2, 1, keepdims=True) + noise), 2, 1)
	data3 = np.dot(S2, np.dot(S1, data3.T)).T

	d = np.sqrt(np.sum(data3**2, 1, keepdims=True))
	d1 = np.reshape(d<2.5, (numOfSamples))
	data3[np.ix_(d1, [True, True])] = data3[np.ix_(d1, [True, True])]/np.repeat(d[d1], 2, 1)

	label3 = np.ones((numOfSamples), dtype=np.uint16) + 1
	label3 = I[label3, :]

	return (data1, label1, data2, label2, data3, label3)

# Devide dataset to training, validating, testing
def devideDataset(data1, label1, data2, label2, data3, label3, numOfTrain, numOfValidation):
	train_X = np.concatenate((data1[0:numOfTrain, :],
	                          data2[0:numOfTrain, :],
	                          data3[0:numOfTrain, :]))

	train_Y = np.concatenate((label1[0:numOfTrain, :],
	                          label2[0:numOfTrain, :],
	                          label3[0:numOfTrain, :]))

	val_X = np.concatenate((data1[numOfTrain:(numOfTrain + numOfValidation), :],
	                          data2[numOfTrain:(numOfTrain + numOfValidation), :],
	                          data3[numOfTrain:(numOfTrain + numOfValidation), :]))

	val_Y = np.concatenate((label1[numOfTrain:(numOfTrain + numOfValidation), :],
	                          label2[numOfTrain:(numOfTrain + numOfValidation), :],
	                          label3[numOfTrain:(numOfTrain + numOfValidation), :]))

	test_X = np.concatenate((data1[(numOfTrain + numOfValidation):, :],
	                        data2[(numOfTrain + numOfValidation):, :],
	                        data3[(numOfTrain + numOfValidation):, :]))

	test_Y = np.concatenate((label1[(numOfTrain + numOfValidation):, :],
	                        label2[(numOfTrain + numOfValidation):, :],
	                        label3[(numOfTrain + numOfValidation):, :]))

	return (train_X, train_Y, val_X, val_Y, test_X, test_Y)

# Visualize dataset
def visualize_data(data1, data2, data3, figure_num):
    if(data1.shape[0] > 1000):
        sizeOfPoint = 10
    else:
        sizeOfPoint = 50

    plt.figure(figure_num)
    plt.axis('equal')
    plt.scatter(data1[:,0], data1[:,1], sizeOfPoint, 'r')
    plt.scatter(data2[:, 0], data2[:, 1], sizeOfPoint, 'g')
    plt.scatter(data3[:, 0], data3[:, 1], sizeOfPoint, 'b')
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$y$', fontsize=15)
    plt.grid()
    # plt.legend(loc=2)
    # plt.show()
    plt.draw()

def find_decision_boundary(X, Y, W1, b1, W2, b2, config):
    num_fake_data = 150
    num_train_sample = X.shape[0]
    num_feature = X.shape[1]
    num_hidden_node = W1.shape[1]
    num_class = Y.shape[1]

    demo_type = config['demo_type']
    activation_function_type = config['activation_function']
    min_X1 = np.min(X[:, 0]) - 0.1
    max_X1 = np.max(X[:, 0]) + 0.1
    min_X2 = np.min(X[:, 1]) - 0.1
    max_X2 = np.max(X[:, 1]) + 0.1

    # Create grid data
    X1 = np.linspace(min_X1, max_X1, num_fake_data)
    X2 = np.linspace(min_X2, max_X2, num_fake_data)
    xv, yv = np.meshgrid(X1, X2)
    xv = xv.flatten().astype(np.float32).reshape((num_fake_data ** 2, 1))
    yv = yv.flatten().astype(np.float32).reshape((num_fake_data ** 2, 1))
    X = np.concatenate((xv, yv), 1)
    if (demo_type == "classifynnkernel"):
        kernel_poly_order = config['kernel_poly_order']
        X = kernel_preprocess(X, kernel_poly_order)

    del xv
    del yv

    a1 = np.dot(X, W1) + b1
    z1 = activation_function(a1, activation_function_type)
    a2 = np.dot(z1, W2) + b2
    # pred = np.repeat(np.argmax(a2, 1).reshape((num_fake_data**2, 1)), 2, 1)
    pred = np.argmax(a2, 1)
    grid0 = X[pred == 0, :]
    grid1 = X[pred == 1, :]
    grid2 = X[pred == 2, :]

    grid0 = grid0.reshape((grid0.shape[0], num_feature))
    grid1 = grid1.reshape((grid1.shape[0], num_feature))
    grid2 = grid2.reshape((grid2.shape[0], num_feature))

    return (grid0, grid1, grid2)

def visualize_decision_grid(data1, data2, data3, figure_num):
    plt.figure(figure_num)
    plt.axis('equal')
    plt.scatter(data1[:, 0], data1[:, 1], 10, c=[1, 0.5, 0.5], marker='+')
    plt.scatter(data2[:, 0], data2[:, 1], 10, c=[0.5, 1, 0.5], marker='+')
    plt.scatter(data3[:, 0], data3[:, 1], 10, c=[0.5, 0.5, 1], marker='+')
    bp = 1
    
def train_draw(X, Y, W1, b1, W2, b2, config, all_cost, i, J):
    # Display the training process

    num_hidden_node = config['num_hidden_node']
    num_train_per_class = config['num_train_per_class']
    train_method = "Neural Network"
    save_img = config['save_img']
    demo_type = config['demo_type']

    lr = config['lr']

    pylab.clf()
    # f = plt.figure(2, figsize=(16, 8))
    f = plt.figure(2, figsize=(12, 6))

    title = '%s with %d hidden nodes, lr = %.4g, %d epoch, cost = %.4g' % (train_method, num_hidden_node, lr, i, J)

    plt.subplot(1, 2, 1)
    [grid1, grid2, grid3] = find_decision_boundary(X, Y, W1, b1, W2, b2, config)

    visualize_decision_grid(grid1, grid2, grid3, 2)

    visualize_data(X[0:num_train_per_class, :],
                   X[num_train_per_class:num_train_per_class * 2, :],
                   X[num_train_per_class * 2:, :],
                   2)
    plt.subplot(1, 2, 2)
    plt.plot(all_cost, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')

    f.suptitle(title, fontsize=15)
    plt.pause(0.1)

    pylab.draw()

def softmax_log_loss(X, Y, bp=False):
    """
    Calculate softmax log loss (aka categorical cross entropy after softmax)
    :type X: 2D numpy array
    :param X: the predictions/labels computed by the network
    :type Y: 2D numpy array
    :param Y: the groundtruth, what we want X to be
    """
    # Perform checking
    assert len(X.shape) == 2, "X should have a shape of (num_sample, num_class)"
    assert len(Y.shape) == 2, "Y should have a shape of (num_sample, num_class)"
    assert (X.shape[0] == Y.shape[0]) and (X.shape[1] == Y.shape[1]), "Predictions and labels should have the same shape"
    n = Y.shape[0]

    if (not bp):
        # Perform feedforward
        # Assume that the second dim is the feature dim
        xdev = X - np.max(X, 1, keepdims=True)
        lsm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        return -np.sum(lsm*Y) / n
    else:
        # Perform backprob and return the derivatives
        xmax = np.max(X, 1, keepdims=True)
        ex = np.exp(X-xmax)
        dFdX = ex/np.sum(ex, 1, keepdims=True)
        dFdX[Y.astype(bool)] = (dFdX[Y.astype(bool)]-1)
        dFdX = dFdX / n
        # dFdX = (dFdX - 1) / n
        return dFdX

def get_grad(X, Y, W1, b1, W2, b2, config):
    activation_function_type = config['activation_function']

    num_train_sample = X.shape[0]
    num_feature = X.shape[1]
    num_hidden_node = W1.shape[1]
    num_class = Y.shape[1]

    a1 = np.dot(X, W1) + b1
    z1 = activation_function(a1, activation_function_type)
    a2 = np.dot(z1, W2) + b2

    # Calculate W2 and b2 gradient
    dJ_da2 = softmax_log_loss(a2, Y, True)
    # print "dJ_da2 = " ,dJ_da2
    # These are basically dJ_da2 but are repeated so we can multiply them with dJ_dW2 and dJ_dz1
    dJ_da2b = np.sum(dJ_da2, 0, keepdims=True)
    # print "dJ_da2b = ", dJ_da2b
    dJ_da2W = np.repeat(dJ_da2.reshape((num_train_sample, 1, num_class)), num_hidden_node, 1)
    # print "dJ_da2W = ", dJ_da2W
    dJ_da2z1 = np.repeat(dJ_da2.reshape((num_train_sample, 1, num_class)), num_hidden_node, 1)
    # print "dJ_da2z1 = ", dJ_da2z1

    da2_dW2 = np.repeat(z1.reshape((num_train_sample, num_hidden_node, 1)), num_class, 2)
    da2_db2 = 1
    da2_dz1 = np.repeat(W2.reshape(1, num_hidden_node, num_class), num_train_sample, 0)

    dJ_dW2 = np.sum(dJ_da2W * da2_dW2, 0)
    dJ_db2 = da2_db2 * dJ_da2b
    dJ_dz1 = np.sum(dJ_da2z1 * da2_dz1, 2)

    # Calculate W1 and b1 gradient
    dJ_dz1_dW1 = np.repeat(dJ_dz1.reshape((num_train_sample, 1, num_hidden_node)), num_feature, 1)
    dz1_da1 = activation_function(a1, activation_function_type, True)
    dz1_da1_W1 = np.repeat(dz1_da1.reshape((num_train_sample, 1, num_hidden_node)), num_feature, 1)
    da1_dW1 = np.repeat(X.reshape((num_train_sample, num_feature, 1)), num_hidden_node, 2)
    da1_db1 = 1

    dJ_dW1 = np.sum(dJ_dz1_dW1 * dz1_da1_W1 * da1_dW1, 0)
    dJ_db1 = np.sum(dJ_dz1 * dz1_da1 * da1_db1, 0, keepdims=True)

    # NumericalGradientCheck(X, Y, W1, b1, W2, b2, dJ_db1)

    return (dJ_dW1, dJ_db1, dJ_dW2, dJ_db2)

if __name__ == '__main__':
	numOfSamples = 1000
	numOfTrain = 500
	numOfValidation = 250
	(data1, label1, data2, label2, data3, label3) = create_data(numOfSamples)
	(train_X, train_Y, val_X, val_Y, test_X, test_Y) = devideDataset(data1, label1, data2, label2, data3, label3,
                                                                             numOfTrain, numOfValidation)

	# Visualize dataset
	visualize_data(data1, data2, data3, 1)

	# Pre-process data
	mean_X = np.mean(train_X, 0, keepdims=True)
	std_X = np.std(train_X, 0, keepdims=True)
	train_X = (train_X - mean_X) / std_X
	val_X = (val_X - mean_X) / std_X
	test_X = (test_X - mean_X) / std_X

	# Parse param from config
	config = {}
	config['demo_type'] = "classifynnsgd"
	config['save_img'] = False
	config['num_epoch'] = 1000
	config['lr'] = 0.8
	config['num_train_per_class'] = numOfTrain
	config['num_hidden_node'] = 3
	config['activation_function'] = 'relu'
	config['display_rate'] = 10 # epochs per display time
	config['momentum'] = 0.9

	lr = config['lr']
	num_epoch = config['num_epoch']
	num_train_per_class = config['num_train_per_class']
	num_hidden_node = config['num_hidden_node']
	momentum_rate = config['momentum']
	display_rate = config['display_rate']
	activation_function_type = config['activation_function']

	num_train_sample = train_X.shape[0]
	num_feature = train_X.shape[1]
	num_class = train_Y.shape[1]
	print "Number of training samples = ",num_train_sample

	# Create a weight matrix of shape (2, num_hidden_node)
	W1 = rng.randn(num_feature, num_hidden_node)
	b1 = rng.randn(1, num_hidden_node)

	# Create output weight
	W2 = rng.randn(num_hidden_node, num_class)
	b2 = rng.randn(1, num_class)

	# Create momentum storage
	W1m = np.zeros_like(W1)
	b1m = np.zeros_like(b1)
	W2m = np.zeros_like(W2)
	b2m = np.zeros_like(b2)

	num_train_sample = 1
	pylab.ion()
	pylab.show()
	all_cost = []
	for i in range(0, num_epoch):
	    # Calculate the loss
	    a1 = np.dot(train_X, W1) + b1
	    # print "a1 = ", a1
	    z1 = activation_function(a1, activation_function_type)
	    # print "z1 = ", z1
	    a2 = np.dot(z1, W2) + b2
	    # print "a2 = ", a2
	    J = softmax_log_loss(a2, train_Y)
	    # print "loss = ", J

	    # Doing backprop
	    print('[Epoch %d] Train loss: %f' % (i, J))

	    dJ_dW1, dJ_db1, dJ_dW2, dJ_db2 = get_grad(train_X, train_Y, W1, b1, W2, b2, config)
	    # NumericalGradientCheck(train_X, train_Y, W1, b1, W2, b2, dJ_db1)

	    W1m = W1m * momentum_rate + lr * dJ_dW1 * lr
	    b1m = b1m * momentum_rate + lr * dJ_db1 * lr
	    W2m = W2m * momentum_rate + lr * dJ_dW2 * lr
	    b2m = b2m * momentum_rate + lr * dJ_db2 * lr

	    W1 = W1 - W1m
	    b1 = b1 - b1m
	    W2 = W2 - W2m
	    b2 = b2 - b2m

	    all_cost.append(J)

	    if (i % display_rate == 0):
	        config['train_method'] = 'sgdm'
	        train_draw(train_X, train_Y, W1, b1, W2, b2, config, all_cost, i, J)

	    bp = 1

	plt.show()