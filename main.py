import mlp
import numpy as np
if __name__ == '__main__':
    np.random.seed(seed=1693214) #we genereate a fixed seed, so our result will always be the same when training the network or splitting the data.
    content_classifier = mlp.MLP(load_model = False)
    content_classifier.train(epochs=500,batch_size=128)
    print("Sucess!")