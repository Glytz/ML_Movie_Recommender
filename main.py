import argparse

import mlp
import numpy as np
import utils
import baseline

if __name__ == '__main__':
    # default mninst parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=int, default=1)  # paramter if we load the model
    parser.add_argument("--train_model", type=int, default=0)  # parameter if we train the model
    parser.add_argument("--use_data_v1", type=int, default=1)  # Parameter depending on which dataset we want to use
    parser.add_argument("--model_path", type=str, default="/model"+str(1)+"/classifier.h5")  # Parameter depending on which dataset we want to use
    parser.add_argument("--is_softmax", type=int, default=0) #if the model use a softmax activation or not
    parser.add_argument("--seed", type=int, default=1693214)  #seed for the random to use
    parser.add_argument("--epochs", type=int, default=500)  #number of epoch to train the model
    parser.add_argument("--batch_size", type=int, default=128)  #size of the batch to train the model with
    param = parser.parse_args()
    np.random.seed(seed=param.seed)  # we genereate a fixed seed, so our result will always be the same when training the network or splitting the data.
    content_classifier = mlp.MLP(use_data_v1=param.use_data_v1, load_model=param.load_model, model_path=param.model_path)
    if param.train_model:
        content_classifier.train(epochs=param.epochs, batch_size=param.batch_size, is_softmax=param.is_softmax)
    mlp_abs_error, mlp_quad_error = content_classifier.test_model(is_soft_max=param.is_softmax)
    # we run the error of the predictions, we will compare it vs mean as a baseline
    print("Mlp Quadratic Error on test set : " + str(mlp_quad_error))
    print("Mlp Abs Error on test set : " + str(mlp_abs_error))
    # baseline errors preds
    baseline_abs_error, baseline_quadratic_error = baseline.baseline_error()
    print("Baseline absolute error : " + str(baseline_abs_error))
    print("Baseline quadratic error : " + str(baseline_quadratic_error))
