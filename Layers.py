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

