import Layers
import numpy as np

class DeepNetwork():

    def __init__(self, layers):
        self.layers = layers
    
    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x


if __name__ == "__main__":
    n_samples, input_dim, output_dim = 2,3,4
    layer_list = [Layers.Linear(input_dim,5),Layers.ReLU(), Layers.Linear(5,output_dim)]
    net = DeepNetwork(layer_list)
    x = np.random.rand(n_samples,input_dim)
    y = net.feed_forward(x)
    print(f"y = {y}")
    print(np.shape(y))