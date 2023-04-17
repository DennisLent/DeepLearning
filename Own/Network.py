import Layers
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
                layer.weight -= learn_rate*layer.weight_gradient
            if hasattr(layer, "bias"):
                layer.bias -= learn_rate*layer.bias_gradient

def MSE(true, prediction):
    loss = np.mean((prediction - true)**2)
    gradient = 2*(prediction - true)
    return loss, gradient

def CrossEntropy(true, prediction):

    def softmax(vec):
        return np.exp(vec) / np.sum(np.exp(vec), axis=1, keepdims=True)
    
    prediction_softmax = softmax(prediction)
    true = np.argmax(true, axis=1)
    num_samples = np.shape(true)[0]
    probability = -np.log(prediction_softmax[np.arange(num_samples), true])
    loss = np.mean(probability)
    gradient = prediction_softmax
    gradient[np.arange(num_samples), true] -= 1
    gradient /= num_samples

    return loss, gradient


if __name__ == "__main__":

    def test_forward():
        n_samples, input_dim, output_dim = 2,3,4
        layer_list = [Layers.Linear(input_dim,5),Layers.ReLU(), Layers.Linear(5,output_dim)]
        net = DeepNetwork(layer_list)
        x = np.random.rand(n_samples,input_dim)
        y = net.feed_forward(x)
        print(f"y = {y}")
        print(np.shape(y))


    def x_or_problem(lossfunc, learning_rate, iterations, title=None):
        x_or = np.array([[0,0], [0,1], [1, 0], [1,1]])
        result = np.array([[1,0], [0,1], [0,1], [1,0]])
        input_dim, hidden_dim, output_dim = 2, 10, 2
        layers = [Layers.Linear(input_dim, hidden_dim), Layers.ReLU(), Layers.Linear(hidden_dim, output_dim)]
        network = DeepNetwork(layers)
        net_loss = []
        net_acc = []

        for _ in tqdm(range(iterations)):
            prediction = network.feed_forward(x_or)
            loss, gradient = lossfunc(result, prediction)
            net_loss.append(loss)
            network.feed_backward(gradient)
            network.optimizer(learning_rate)

            true_labels = np.argmax(result, axis=1)
            predicted_labels = np.argmax(prediction, axis=1)
            accuracy = np.mean(predicted_labels == true_labels)
            net_acc.append(accuracy)
        
        fig, ax = plt.subplots(ncols=3, figsize=(15,15))
        ax[0].plot(net_loss, label="loss")
        ax[0].set_xlabel("Iterations")
        ax[0].set_title("Loss")
        ax[1].plot(net_acc, label="accuracy")
        ax[1].set_xlabel("Iterations")
        ax[1].set_title("Accuracy")

        xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        prediction = network.feed_forward(grid)
        y = np.argmax(result, axis=1)
        Z = np.argmax(prediction, axis=1).reshape(xx.shape)

        ax[2].scatter(x_or[:, 0], x_or[:, 1], c=y, cmap='coolwarm')
        ax[2].contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        ax[2].set_title("Contour Plot")
        if title is not None:
            fig.suptitle(title)

        plt.show()


    #test_forward()
    x_or_problem(MSE, 1e-2, 100, "XOR Problem with MSE")
    x_or_problem(CrossEntropy, 5e-1, 150, "XOR Problem with Cross-Entropy")