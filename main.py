import mlp
import numpy as np
import utils
import baseline

if __name__ == '__main__':
    load_model = True
    train_model = False
    use_data_v1 = True
    modelnumber = 2
    np.random.seed(
        seed=1693214)  # we genereate a fixed seed, so our result will always be the same when training the network or splitting the data.
    content_classifier = mlp.MLP(use_data_v1=use_data_v1, load_model=load_model, model_path="/model"+str(modelnumber)+"/classifier.h5")
    if (train_model):
        content_classifier.train(epochs=500, batch_size=128)
    mlp_abs_error, mlp_quad_error = content_classifier.test_model(is_soft_max=True)
    # we run the error of the predictions, we will compare it vs mean as a baseline
    print("Mlp Quadratic Error on test set : " + str(mlp_quad_error))
    print("Mlp Abs Error on test set : " + str(mlp_abs_error))
    # baseline errors preds
    baseline_abs_error, baseline_quadratic_error = baseline.baseline_error()
    print("Baseline absolute error : " + str(baseline_abs_error))
    print("Baseline quadratic error : " + str(baseline_quadratic_error))
