# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pylab
import time
import matplotlib.text as plttext
import itertools
import collections
rng = np.random.RandomState(1311)
import pdb
# %matplotlib inline

# Activation functions
def sigmoid(X, bp = False):
    if (not bp):
        return 1.0/(1.0 + np.exp(-X))
    else:
        sig = 1.0/(1.0 + np.exp(-X))
        return sig * (1 - sig)

def relu(X, bp = False):
    result = X
    if (not bp):
        result = X
        result[X < 0] = 0
    else:
        result[X > 0] = 1
        result[X <= 0] = 0
    return result

def softmax(X):
    # Assume that the second dim is the feature dim
    max_input = np.max(X, 1, keepdims=True)
    X_max = X - max_input
    e = np.exp(X_max)
    sum_e = np.sum(e, 1, keepdims=True)
    return e / sum_e

def logistic(z):
    return 1 / (1 + np.exp(-z))

def logistic_deriv(y):  # Derivative of logistic function
    return np.multiply(y, (1 - y))

# Define the cost function
def cost(y, t):
    # pdb.set_trace()
    return 0.5 * ((t - y)**2).sum()

def activation_function(X, type, bp = False):
    if (type == "sigmoid"):
        return sigmoid(X, bp)
    elif (type == "relu"):
        return relu(X, bp)
    else:
        raise ValueError("Activation function not recognized")

# Define layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""
    
    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []
    
    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in this case).
        output_grad is the gradient at the output of this layer 
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the 
         output error instead of output_grad"""
        pass

class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""
    
    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
        
    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                                np.nditer(self.b, op_flags=['readwrite']))
    
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)

class NonLinearLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return logistic(X)
        # return activation_function(X, 'sigmoid')
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        # print "dev at NonLinear Layer = ", np.multiply(activation_function(Y, 'sigmoid', True), output_grad)
        # return np.multiply(activation_function(Y, 'sigmoid', True), output_grad)
        return np.multiply(logistic_deriv(Y), output_grad)

class OutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        # return softmax(X)
        return X
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        # print "dev at Output Layer = ", (Y - T)
        return (Y - T) #/ Y.shape[0]
    
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        # return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]
        return cost(Y, T)

# Define the forward propagation step as a method
def forward_step(input_samples, layers):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.  
    """
    activations = [input_samples] # List of layer acctivations
    # Compute the forward activations for each layer starting from the first
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X) # Get the output of the current layer
        activations.append(Y)
        X = activations[-1]
    return activations

# Define the backward propagation step as a method
def backward_step(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.
    Input:
        activations: A list of forward step activations where the activation at 
            each index i+1 corresponds to the activation of layer i in layers. 
            activations[0] contains the input samples. 
        targets: The output targets of the output layer.
        layers: A list of Layers corresponding that generated the outputs in activations.
    Output:
        A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers. 
    """
    param_grads = collections.deque() # List of parameter gradients for each layer
    output_grad = None # The error gradient at the output of the current layer
    # Propagate the error backwards through all the layers.
    # Use reversed to iterate backwards over the list of layers.
    for layer in reversed(layers):
        Y = activations.pop() # Get the activations of the last layer on the stack
        # print "activation.pop()"
        # print Y
        # Compute the error at the output layer
        # The output layer error is calculated different then hidden layer error
        if output_grad is None:
            # print "input_grad = layer.get_input_grad(Y, targets)"
            # print "return (Y - T)"
            input_grad = layer.get_input_grad(Y, targets)
        else:
            # print "input_grad = layer.get_input_grad(Y, output_grad)"
            input_grad = layer.get_input_grad(Y, output_grad)
            
        # Get the input of this layer (activations of the previous layer)
        X = activations[-1]
        # print "activation of prev layer"
        # print X

        # Compute the layer parameter gradients used to update the parameters
        grads = layer.get_params_grad(X, output_grad)
        # print "gradient of W, b"
        # print grads
        param_grads.appendleft(grads)
        # Compute gradient at output of previous layer (input of current layer)
        output_grad = input_grad
        # print "output_grad"
        # print output_grad
        # print "-----------------------"
    return list(param_grads)


# Define a method to update the parameters
def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in itertools.izip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter

# Define gradient checking
def gradient_checking(layers, X_train, T_train, oneD=False):
    # Perform gradient checking
    # print "++++++++ checking gradient ++++++++++"
    nb_samples_gradientcheck = 5 # Test the gradients on a subset of the data
    if oneD == False:
        X_temp = X_train[0:nb_samples_gradientcheck,:]
        T_temp = T_train[0:nb_samples_gradientcheck,:]
    else:
        X_temp = X_train[0:nb_samples_gradientcheck].reshape((nb_samples_gradientcheck,1))
        T_temp = T_train[0:nb_samples_gradientcheck].reshape((nb_samples_gradientcheck,1))

    # print "dataset X:"
    # print X_temp
    # print "dataset Y:"
    # print T_temp

    # Get the parameter gradients with backpropagation
    activations = forward_step(X_temp, layers)
    # print "activations = ", len(activations)
    # for i in range(len(activations)):
    #     print "activation ", i
    #     print activations[i]

    param_grads = backward_step(activations, T_temp, layers)

    # for i in range(len(layers)):
    #     print "\tparam ",i
    #     print param_grads[i]

    # print "+++++++++++++++++++++++++++++++++++++++"
    # print "param_grads = ", len(param_grads)
    # print param_grads[3]

    # Set the small change to compute the numerical gradient
    eps = 0.0001

    # Compute the numerical gradients of the parameters in all layers
    for idx in range(len(layers)):
        # print "layer ", idx
        layer = layers[idx]
        layer_backprop_grads = param_grads[idx]

        # Compute the numerical gradient for each parameter in the layer
        for p_idx, param in enumerate(layer.get_params_iter()):
            grad_backprop = layer_backprop_grads[p_idx]
            # + eps
            param += eps
            plus_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # - eps
            param -= 2 * eps
            min_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # reset param value
            param += eps
            # calculate numerical gradient
            grad_num = (plus_cost - min_cost)/(2*eps)
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, grad_backprop):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_backprop)))
    print('No gradient errors found')

