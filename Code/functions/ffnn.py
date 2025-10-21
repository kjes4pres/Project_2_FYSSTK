import autograd.numpy as np  
from autograd import grad



class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        optimization_method=None,
    ):
        self.cost_der = cost_der
        self.cost_fun = cost_fun
        self.activation_ders = activation_ders
        self.activation_funcs = activation_funcs
        self.layer_output_sizes = layer_output_sizes
        self.network_input_size = network_input_size
        self.weights = self.create_layers(network_input_size, layer_output_sizes)
        self.optimization_method = optimization_method
        self.training_info = {
            "Cost_history" : []
            }

    def get_weights(self):
        return self.weights
    
    def get_info(self):
        return self.training_info
    
    def create_layers(self, network_input_size, layer_output_sizes):
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size, )
            layers.append((W, b))
            i_size = layer_output_size

        return layers
    
    def _feed_forward(self, input):
        a = input
        for (W, b), activation_func in zip(self.weights, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def cost(self, input, target):
        predict = self._feed_forward(input)
        return self.cost_fun(predict, target)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs

        for (W, b), activation_func in zip(self.weights, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a



    def backpropagation_batch(self, inputs, target):

        layer_inputs, zs, predictions = self._feed_forward_saver(inputs)
        batch_size = inputs.shape[0]

        layer_grads = [() for l in self.weights]

        # Loop over layers backward
        for i in reversed(range(len(self.weights))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == len(self.weights) - 1:
                # Last layer: derivative of cost w.r.t activation
                dC_da = self.cost_der(predictions, target) / batch_size
            else:
                W, _ = self.weights[i + 1]
                dC_da = dC_dz @ W.T
        
            dC_dz = dC_da * activation_der(z)
            dC_dW = (layer_input.T @ dC_dz) 
            dC_db = np.mean(dC_dz, axis=0)
            

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    # Define weight update methods based on optimization method


    def update_weights_RMSProp(self, layer_grads, learning_rate, beta=0.9, epsilon=1e-8):
        velocities = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.weights]
        for i in range(len(self.weights)):
            W, b = self.weights[i]
            dC_dW, dC_db = layer_grads[i]
            vW, vb = velocities[i]
            vW = beta * vW + (1 - beta) * (dC_dW ** 2)
            vb = beta * vb + (1 - beta) * (dC_db ** 2)
            W -= (learning_rate / (np.sqrt(vW) + epsilon)) * dC_dW
            b -= (learning_rate / (np.sqrt(vb) + epsilon)) * dC_db
            self.weights[i] = (W, b)

    

    def update_weights_Adam(self, layer_grads, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):

        m = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.weights]
        v = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.weights]

        for i in range(len(self.weights)):
            W, b = self.weights[i]
            dC_dW, dC_db = layer_grads[i]

            m[i, 0] = beta1 * m[i, 0] + (1 - beta1) * dC_dW
            v[i, 0] = beta2 * v[i, 0] + (1 - beta2) * (dC_dW ** 2)

            m[i, 1] = beta1 * m[i, 1] + (1 - beta1) * dC_db
            v[i, 1] = beta2 * v[i, 1] + (1 - beta2) * (dC_db ** 2)

            m_W_hat = m[i, 0] / (1 - beta1 ** (t + 1))
            v_W_hat = v[i, 0] / (1 - beta2 ** (t + 1))

            m_b_hat = m[i, 1] / (1 - beta1 ** (t + 1))
            v_b_hat = v[i, 1] / (1 - beta2 ** (t + 1))

            W -= (learning_rate / (np.sqrt(v_W_hat) + epsilon)) * m_W_hat
            b -= (learning_rate / (np.sqrt(v_b_hat) + epsilon)) * m_b_hat
            self.weights[i] = (W, b)

        t = t + 1

        return t

    
        # Standard gradient descent
    def update_weights(self, layer_grads, learning_rate):
        for i in range(len(self.weights)):
            W, b = self.weights[i]
            dC_dW, dC_db = layer_grads[i]

            W -= learning_rate * dC_dW
            b -= learning_rate * dC_db

            self.weights[i] = (W, b)
    
    def train(self, input, target, epochs = 1000, learning_rate = 0.1):
        for i in range(epochs):
            grads = self.backpropagation_batch(input,target)
            self.update_weights(grads, 0.1)
            self.training_info["Cost_history"].append(self.cost(input,target))

    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        auto_grad = grad(self.cost_fun, 0)
        return auto_grad(inputs, targets)