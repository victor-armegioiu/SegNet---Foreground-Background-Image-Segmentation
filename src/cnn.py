import numpy as np
# from get_data import DataBatch
from math import ceil, floor
from im2col import *
from get_data import DataBatch
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class LayerInterface(object):
    def __init__(self, inputs_no, outputs_no, transfer_function):
        self.outputs = np.array([])

    def forward(self, inputs):
        return self.outputs

    def backward(self, inputs, output_errors):
        return None

    def update_parameters(self, learning_rate):
        pass

    def to_string(self):
        pass


class ConvLayer(LayerInterface):
    def __init__(self, inputs_depth, inputs_size, outputs_depth, k, stride):
        # Number of inputs, number of outputs, filter size, stride

        self.inputs_depth = inputs_depth
        self.inputs_size = inputs_size

        self.k = k
        self.stride = stride

        self.outputs_depth = outputs_depth
        self.outputs_height = int((self.inputs_size - self.k) / self.stride + 1)
        self.outputs_width = int((self.inputs_size - self.k) / self.stride + 1)

        # Layer's parameters
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + self.inputs_depth + self.k + self.k)),
            (self.outputs_depth, self.inputs_depth, self.k, self.k)
        )
        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + 1)),
            (self.outputs_depth, 1)
        )

        # Computed values
        self.outputs = np.zeros((self.outputs_depth, self.outputs_height, self.outputs_width))

        # Gradients
        # self.g_weights = np.zeros(self.weights.shape)
        # self.g_biases = np.zeros(self.biases.shape)

        # Padding
        outputs_size = ceil(self.inputs_size / self.stride)
        self.padding = (self.inputs_size - 1) * self.stride + self.k - self.inputs_size

        self.padding_top = floor(self.padding / 2)
        self.padding_bottom = self.padding - self.padding_top

        self.last_input_shape = None
        self.cache = None

        self.dX = None
        self.dW = None
        self.db = None

    def forward(self, inputs):
        cache = self.weights, self.biases, self.stride, self.padding

        n_x, d_x, h_x, w_x = inputs.shape
        n_filters, d_filter, h_filter, w_filter = self.weights.shape
        h_out = (h_x - h_filter + self.padding) / self.stride + 1
        w_out = (w_x - w_filter + self.padding) / self.stride + 1
      
        inputs = np.pad(inputs, ((0, 0), (0, 0), (self.padding_top, self.padding_bottom), (self.padding_top, self.padding_bottom)), mode='constant')
        self.last_input_shape = inputs.shape

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        X_col = im2col_indices(inputs, h_filter, w_filter, padding=0, stride=self.stride)
        W_col = self.weights.reshape(n_filters, -1)

        out = np.dot(W_col, X_col) + self.biases
        out = out.reshape(n_filters, h_out, w_out, n_x)
       
        out = out.transpose(3, 0, 1, 2)
       
        cache = (inputs, self.weights, self.biases, self.stride, self.padding, X_col)
        self.cache = cache

        return out

    def backward(self, dout):
        cache = self.cache
        X, W, b, stride, padding, X_col = cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = np.dot(dout_reshaped, X_col.T)
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = np.dot(W_reshape.T, dout_reshaped)

        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=0, stride=stride)
        dX = dX[:, :, self.padding_top : -self.padding_bottom, self.padding_top : -self.padding_bottom]

        self.dX = dX
        self.dW = dW
        self.db = db

        return dX

    def update_parameters(self, learning_rate):
        if not hasattr(self, 'm_t_w'):
            self.m_t_w = np.zeros_like(self.weights)
            self.v_t_w = np.zeros_like(self.weights)
            self.m_t_b = np.zeros_like(self.biases)
            self.v_t_b = np.zeros_like(self.biases)
            self.t = 1
        else: self.t += 1

        alpha = learning_rate
        beta_1, beta_2 = 0.9, 0.999
        epsilon = 1e-8

        g_t = self.dW
        self.m_t_w = beta_1*self.m_t_w + (1-beta_1)*g_t  
        self.v_t_w = beta_2*self.v_t_w + (1-beta_2)*(g_t*g_t) 
        m_cap = self.m_t_w / (1-(beta_1**self.t))    
        v_cap = self.v_t_w / (1-(beta_2**self.t))                                
        self.weights = self.weights - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)   

        g_t = self.db
        self.m_t_b = beta_1*self.m_t_b + (1-beta_1)*g_t  
        self.v_t_b = beta_2*self.v_t_b + (1-beta_2)*(g_t*g_t)
        m_cap = self.m_t_b / (1-(beta_1**self.t))     
        v_cap = self.v_t_b / (1-(beta_2**self.t))                             
        self.biases = self.biases - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)  


def binary_cross_entropy_loss(my_preds, gt_preds):
    assert my_preds.shape == gt_preds.shape

    batch_size = my_preds.shape[0]
    IMG_SIZE = my_preds.shape[1]

    eps = 0.00001

    my_preds[my_preds == 0.0] = eps
    my_preds[my_preds == 1.0] = 1.0 - eps

    loss = - (gt_preds * np.log(my_preds))
    loss -= (1. - gt_preds) * np.log(1. - my_preds)
    loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)

    return loss


def softmax(X):
    max_val = np.max(X, axis=1)
    ex = np.exp(X - max_val[:, np.newaxis])
    probs = ex / np.sum(ex, keepdims=True, axis=1)
    return probs


class ReluLayer(LayerInterface):
    def __init__(self):
        self.cache = None
        self.dx = None

    def forward(self, x):
        out = None

        relu = lambda x : x * (x > 0).astype(float)
        out = relu(x)

        self.cache = x
        return out

    def backward(self, dout):
        dx, x = None, self.cache
        dx = dout * (x >= 0)
        self.dx = dx
        return dx


class Sequential(object):
    def __init__(self):
        self._layers = []
        self.losses = []

    def add(self, layer):
        self._layers.append(layer)

    def forward(self, input_batch):
        for i in range(len(self._layers)):
            input_batch = self._layers[i].forward(input_batch)
        return input_batch

    def backward(self, dout):
        for i in range(len(self._layers) - 1, -1, -1):
            dout = self._layers[i].backward(dout)

    def fit(self, X_train, y_train, learning_rate=0.01, batch_size=50, IMG_SIZE=50):
        output = self.forward(X_train)
        train_probs = softmax(output)

        Y_reshaped = np.reshape(y_train, (-1,)).astype(int)

        num_classes = 2

        gt_probs = np.zeros((len(Y_reshaped), num_classes))
        gt_probs[range(len(Y_reshaped)), Y_reshaped] = 1

        gt_probs = np.reshape(gt_probs, (batch_size, IMG_SIZE, IMG_SIZE, num_classes))
        gt_probs = np.transpose(gt_probs, (0, 3, 1, 2))

        loss = binary_cross_entropy_loss(train_probs[:, 0, :, :], gt_probs[:, 0, :, :])

        self.losses.append(loss)
        print('curr loss is: ', np.mean(loss))

        plt.plot(self.losses)
        plt.savefig('loss.png')

        dout = train_probs - gt_probs
        self.backward(dout)

        for i in range(len(self._layers)):
            self._layers[i].update_parameters(learning_rate)

    def predict(self, X):
        return self.forward(X)


class ResidualBlock(object):
    def __init__(self, layers):
        self._layers = layers

    def forward(self, input_batch):
        original_input = np.copy(input_batch)

        for i in range(len(self._layers) - 1):
            input_batch = self._layers[i].forward(input_batch)

        input_batch = self._layers[-1].forward(input_batch + original_input)
        return input_batch

    def backward(self, dout):
        first_error = self._layers[-1].backward(dout)
        dout = np.copy(first_error)

        for i in range(len(self._layers) - 2, -1, -1):
            dout = self._layers[i].backward(dout)

        return dout + first_error

    def update_parameters(self, learning_rate=0.01):
        for i in range(len(self._layers)):
            self._layers[i].update_parameters(learning_rate)


if __name__ == '__main__':

    # inputs_depth, inputs_size, outputs_depth, k, stride
    model = Sequential()

    model.add(ConvLayer(3, 50, 16, 3, 1))
    model.add(ResidualBlock([ConvLayer(16, 50, 16, 3, 1), ReluLayer(), ConvLayer(16, 50, 64, 3, 1)]))
    model.add(ReluLayer())
    model.add(ConvLayer(64, 50, 2, 3, 1))


    data_batch = DataBatch(train=True, image_dim=50)

    for _ in range(520):
        X_batch, y_batch = data_batch.get_next_batch(batch_size=10)
        model.fit(X_batch, y_batch, batch_size=10)
