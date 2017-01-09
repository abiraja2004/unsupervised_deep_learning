import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt



def load_data():
    """
    load training data and normalize
    """
    digits = load_digits()
    data = np.asarray(digits.data, dtype='float32')
    training_data = data / np.max(data)
    return training_data


def build_model(training_data):
    """
    build and train the rbm.
    """
    rbm = BernoulliRBM(random_state=0, verbose=True, n_components=100,
                       n_iter=50)
    rbm.fit(training_data)
    return rbm


def main():
    rbm = build_model(load_data())
    print(rbm.components_[0].shape)
    # visualize
    plt.figure(figsize=(5, 4.5))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


if __name__ == '__main__':
    main()
