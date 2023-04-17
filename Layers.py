import numpy as np

class Linear():

    def __init__(self, input_dimension, output_dimension):
        self.weight = np.random.normal(loc=0.0, scale=0.1, size=(input_dimension, output_dimension))
        self.bias = np.random.normal(loc=0.0, scale=0.1, size=(output_dimension))
    
    def feed_forward(self, x):
        y = (x @ self.weight) + self.bias
        self.cache = x
        return y
    
    def feed_backward(self, dup):
        x = self.cache
        dx = dup @ self.weight.transpose()
        self.weight_gradient = x.transpose() @ dup
        self.bias_gradient = np.sum(dup, axis=0)

        return dx
    
    def __str__(self) -> str:
        return f"Linear Layer: w = {np.shape(self.weight)} b = {np.shape(self.bias)}"

class ReLU():
    
    def feed_forward(self, x):
        y = np.maximum(0, x)
        self.cache = y
        return y
    
    def feed_backward(self, dup):
        dx = dup.copy()
        y = self.cache
        dx[y == 0] = 0
        return dx

    def __str__(self) -> str:
        return "ReLU"

class LeakyReLU():
    
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def feed_forward(self, x):
        y = np.maximum(self.cutoff*x, x)
        self.cache = y
        return y
    
    def feed_backward(self, dup):
        dx = dup.copy()
        y = self.cache
        dx[y<=0] = self.cutoff*dx[y<=0]
        return dx

    def __str__(self) -> str:
        return f"Leaky ReLU cut = {self.cutoff}"

class Sigmoid():

    def feed_forward(self, x):
        y = 1/(1 + np.exp(-x))
        self.cache = y
        return y
    
    def feed_backward(self, dup):
        y = self.cache
        dx = y*(1-y)*dup
        return dx

    def __str__(self) -> str:
        return "Sigmoid"

class Tanh():

    def feed_forward(self, x):
        y = np.tanh(x)
        self.cache = y
        return y
    
    def feed_backward(self, dup):
        y = self.cache
        dx = (1-np.power(y, 2))*dup
        return dx
    
    def __str__(self) -> str:
        return "Tanh"

class Softmax():
    
    def feed_forward(self, x):
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.cache = y
        return y

    def feed_backward(self, dup):
        n = np.shape(dup)[0]
        dx = self.cache
        dx[np.arange(n), np.argmax(dup, axis=1)] -= 1
        dx /= n
        return dx
    
    def __str__(self) -> str:
        return "Softmax"

class Conv2D():

    def __init__(self, input_dimension, output_dimension, kernel, stride, padding):
        self.in_dim = input_dimension
        self.out_dim = output_dimension
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.random.normal(loc=0.0, scale=0.1, size=(output_dimension, input_dimension, self.kernel, self.kernel))
        self.bias = np.random.normal(loc=0.0, scale=0.1, size=(output_dimension))
    
    def feed_forward(self, x):
        padded = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding)), mode="constant")
        n, _, h, w = np.shape(x)
        hp, wp = 1 + (h + 2*self.padding - self.kernel) // self.stride, 1 + (w + 2*self.padding - self.kernel) // self.stride
        y = np.empty(shape=(n, self.out_dim, hp, wp))
        for i in range(hp):
            for j in range(wp):
                h_diff = i*self.stride
                w_diff = j*self.stride

                output = padded[:,:, h_diff:h_diff+self.kernel, w_diff:w_diff+self.kernel]

                y[:,:,i,j] = (output * self.weight.reshape(self.out_dim, -1, 1, 1)).sum(axis=(2,3)) + self.bias

        self.cache = x
        return y
    
    def feed_backward(self, dup):
        x = self.cache
        dx = np.zeros_like(x)
        self.weight_gradient = np.zeros_like(self.weight)
        n, _, hp, wp = np.shape(dup)

        for i in range(hp):
            for j in range(wp):
                h_diff = i*self.stride
                w_diff = j*self.stride

                output = x[:,:,h_diff:h_diff+self.kernel,w_diff:w_diff+self.kernel]
                dwindow = dx[:,:,h_diff:h_diff+self.kernel,w_diff:w_diff+self.kernel]

                for k in range(n):
                    dwindow[k] += np.sum(self.weight * dup[k, :, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    self.weight_grad += output[k][np.newaxis, :, :, :] * dup[k, :, i, j][:, np.newaxis, np.newaxis, np.newaxis]

        H = x.shape[2] - 2 * self.padding
        W = x.shape[3] - 2 * self.padding

        dx = dx[:, :, self.padding:self.padding+H, self.padding:self.padding+W]

        self.bias_grad = np.sum(dup, axis=(0, 2, 3))

        return dx
    
    def __str__(self) -> str:
        return f"2D Conv k={self.kernel}x{self.kernel}"

class MaxPool2D():

    def __init__(self, kernel, stride, padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
    
    def feed_forward(self, x):
        padded = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding)), mode="constant")
        n, c, h, w = np.shape(x)

        hp, wp = 1 + (h + 2*self.padding - self.kernel) // self.stride, 1 + (w + 2*self.padding - self.kernel) // self.stride

        y = np.empty(shape=(n*c, hp, wp))

        for i in range(hp):
            for j in range(wp):
                h_diff = i*self.stride
                w_diff = j*self.stride

                output = padded[:,:,h_diff:h_diff+self.kernel,w_diff:w_diff+self.kernel].reshape(n*c, -1)

                y[:,i,j], _ = np.max(output, axis=1)
        
        y = y.reshape(n, c, hp, wp)
        self.cache = x

        return y
    
    def feed_backward(self, dup):
        x = self.cache
        dx = np.zeros_like(x)

        n, c, hp, wp = np.shape(dx)

        for i in range(hp):
            for j in range(wp):
                h_diff = i * self.stride
                w_diff = j * self.stride
                window = x[:, :, h_diff:h_diff+self.kernel, w_diff:w_diff+self.kernel].reshape(n*c, -1)

                indices = np.argmax(window, axis=1)

                dwindow = np.zeros_like(window)

                dwindow[np.arange(n*c), indices] += dup[:, :, i, j].ravel()

                dx[:, :, h_diff:h_diff+self.kernel, w_diff:w_diff+self.kernel] += dwindow.reshape(n, c, self.kernel, self.kernel)
        
        return dx

    def __str__(self) -> str:
        return f"2D Max Pooling"




        

if __name__ == "__main__":
    
    def test_backward():
        n_samples, input_dim, output_dim = 2,3,4
        layer = Linear(input_dim, output_dim)
        x = np.random.rand(n_samples,input_dim)
        dy = np.random.rand(n_samples, output_dim)
        y = layer.feed_forward(x)
        dx = layer.feed_backward(dy)
        print(f"y = {y} of shape: {np.shape(y)}")
        print(f"dx = {dx} of shape: {np.shape(dx)}")
    
    def test_attributes():
        layers = [Linear(3,2), ReLU(), Linear(5,2), Sigmoid()]
        for layer in layers:
            if hasattr(layer, "weight"):
                print("it has")
    
    
    test_backward()
    test_attributes()

