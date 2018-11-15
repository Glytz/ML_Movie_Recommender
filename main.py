import mlp
if __name__ == '__main__':
    content_classifier = mlp.MLP(load_model = False)
    content_classifier.train(epochs=5,batch_size=128)
    print("Sucess!")