import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad


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

    def get_weights(self):
        return self.weights

    def feed_forward(self, input, layers, activation_funcs):
        a = input
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def cost(self, layers, input, activation_funcs, target):
        predict = feed_forward(input, layers, activation_funcs)
        return self.cost_fun(predict, target)

    def _feed_forward_saver(self, inputs, layers, activation_funcs):
        layer_inputs = []
        zs = []
        a = inputs

        for (W, b), activation_func in zip(layers, activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def create_layers(self, network_input_size, layer_output_sizes):
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size, 1)
            layers.append((W, b))
            i_size = layer_output_size

        return layers

    def backpropagation_batch(
        self, inputs, layers, activation_funcs, target, activation_ders, cost_der
    ):

        layer_inputs, zs, predictions = self._feed_forward_saver(
            inputs, layers, activation_funcs
        )
        batch_size = inputs.shape[1]

        layer_grads = [() for l in layers]

        # Loop over layers backward
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

            if i == len(layers) - 1:
                # Last layer: derivative of cost w.r.t activation
                dC_da = cost_der(predictions, target) / batch_size
            else:
                W_next, b_next = layers[i + 1]
                dC_da = W_next @ dC_dz

            dC_dz = dC_da * activation_der(z)
            dC_dW = (layer_input @ dC_dz.T) / batch_size
            dC_db = np.mean(dC_dz, axis=1, keepdims=True)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads
    
    # Define weight update methods based on optimization method
    if self.optimization_method == "RMSProp":

        def update_weights(self, layer_grads, learning_rate, beta=0.9, epsilon=1e-8):
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

    elif self.optimization_method == "Adam":

        def update_weights(self, layer_grads, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):

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

    elif self.optimization_method is None:
        # Standard gradient descent
        def update_weights(self, layer_grads, learning_rate):
            for i in range(len(self.weights)):
                W, b = self.weights[i]
                dC_dW, dC_db = layer_grads[i]

                W -= learning_rate * dC_dW
                b -= learning_rate * dC_db

                self.weights[i] = (W, b)

    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        auto_grad = grad(self.cost_fun, 0)
        return auto_grad(inputs, targets)
