import numpy as np

class Linear():

    def __init__(self, input_dimension, output_dimension):
        self.weight = np.random.normal(loc=0.0, scale=0.1, size=(input_dimension, output_dimension))
        self.bias = np.random.normal(loc=0.0, scale=0.1, size=(output_dimension))
    
    def feed_forward(self, x):
        y = (x @ self.weight) + self.bias
        return y
    
    def __str__(self) -> str:
        return f"Linear Layer: w = {np.shape(self.weight)} b = {np.shape(self.bias)}"

class ReLU():
    
    def feed_forward(self, x):
        y = np.maximum(0, x)
        return y

    def __str__(self) -> str:
        return "ReLU"

class LeakyReLU():
    
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def feed_forward(self, x):
        y = np.maximum(self.cutoff, x)
        return y

    def __str__(self) -> str:
        return f"Leaky ReLU cut = {self.cutoff}"

class Sigmoid():

    def feed_forward(self, x):
        y = 1/(1 + np.exp(-x))
        return y

    def __str__(self) -> str:
        return "Sigmoid"

class Tanh():

    def feed_forward(self, x):
        y = np.tanh(x)
    
    def __str__(self) -> str:
        return "Tanh"

if __name__ == "__main__":
    n_samples, in_features, out_features = 2, 3, 4
    x = np.random.rand(n_samples, in_features)

    layer = Linear(in_features, out_features)
    print(np.shape(x))
    print(layer)
    y = layer.feed_forward(x)

    print('Shape of ouput tensor y:', np.shape(y))
