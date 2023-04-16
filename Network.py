import Layers
import numpy as np

class DeepNetwork():

    def __init__(self, layers):
        self.layers = layers
    
    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x
    
    def feed_backward(self, dx):
        for layer in reversed(self.layers):
            dx = layer.feed_backward(dx)
        return dx
    
    def optimizer(self, learn_rate):
        for layer in self.layers:
            if hasattr(layer, "weight"):
                layer.weight = layer.weight - learn_rate*layer.weight_gradient
            if hasattr(layer, "bias"):
                layer.bias = layer.bias - learn_rate*layer.bias_gradient


if __name__ == "__main__":

    def test_forward():
        n_samples, input_dim, output_dim = 2,3,4
        layer_list = [Layers.Linear(input_dim,5),Layers.ReLU(), Layers.Linear(5,output_dim)]
        net = DeepNetwork(layer_list)
        x = np.random.rand(n_samples,input_dim)
        y = net.feed_forward(x)
        print(f"y = {y}")
        print(np.shape(y))        
        
    
    test_forward()