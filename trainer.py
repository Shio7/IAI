import argparse

import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
from keras.optimizers import Adam
# from keras import backend as K


class Data:
    """
    Define dataset for training GAN
    """
    def __init__(self, mu, sigma, input_dim):
        """
        ex) real_sample(n) -> (n, input_dim) data generate
        :param mu: mean of data
        :param sigma: sigma of data
        :param input_dim: input dimension
        """
        self.real_sample = lambda n_batch: np.random.normal(mu, sigma, (n_batch, input_dim))
        self.z_sample = lambda n_batch: np.random.rand(n_batch, input_dim)


optimizer = Adam(lr=2e-4, beta_1=0.9, beta_2=0.999)


class GAN:
    """
    GAN 모델 정의
    """
    def __init__(self, learning_rate, input_dim):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.D = self.discriminator()
        self.G = self.generator()
        self.GD = self.makeGD()

    def discriminator(self):
        D = models.Sequential()
        D.add(Dense(50, activation='relu', input_shape=(self.input_dim,))) # np.shape(100) == np.shape(100,)과 같은것
        D.add(Dense(50, activation='relu'))
        D.add(Dense(1, activation='sigmoid'))
        D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return D

    def generator(self):
        G = models.Sequential()
        G.add(Reshape((self.input_dim, 1), input_shape=(self.input_dim,)))
        G.add(Conv1D(50, 1, activation='relu'))
        G.add(Conv1D(50, 1, activation='sigmoid'))
        G.add(Conv1D(1, 1))
        G.add(Flatten())
        G.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return G

    def makeGD(self):
        G, D = self.G, self.D
        GD = models.Sequential()
        GD.add(G)
        GD.add(D)
        D.trainable = False
        GD.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        D.trainable = True
        return GD


class Model:
    """
    Learner class
    """
    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.data = Data(args.mu, args.sigma, args.input_dim)

        # the reason why D, G differ in iter : Generator needs more training than Discriminator
        self.n_iter_D = args.n_iter_D
        self.n_iter_G = args.n_iter_G
        self.gan = GAN(self.learning_rate, args.input_dim)

    def fit(self):
        """
        train model while epoch
        """
        for epoch in range(self.epochs):
            # train discriminator by real data
            for iter in range(self.n_iter_D):
                self.train_D()

            # train GD by generated fake data
            for iter in range(self.n_iter_G):
                self.train_GD()

            # print and show each 10n epoch
            if (epoch + 1) % 10 == 0:
                gan = self.gan
                data = self.data
                z = data.z_sample(100)
                gen = gan.G.predict(z)
                real = data.real_sample(100)
                self.show_hist(real, gen, z)
                self.print_stats(real, gen)


    def train_D(self):
        """
        train Discriminator
        """

        # Real data
        real = self.data.real_sample(self.batch_size)

        # Generated data
        z = self.data.z_sample(self.batch_size)
        gen = self.gan.G.predict(z)

        # train discriminator
        self.gan.D.trainable = True
        x = np.concatenate([real, gen], axis=0)
        y = np.array([1] * real.shape[0] + [0] * gen.shape[0])
        self.gan.D.train_on_batch(x, y)

    def train_GD(self):
        """
        train Generator (Not Discriminator)
        """

        # seed data for data generation
        z = self.data.z_sample(self.batch_size)

        # only train generator
        self.gan.D.trainable = False

        # train generator
        y = np.array([1] * z.shape[0])
        self.gan.GD.train_on_batch(z, y)

    def show_hist(self, real, gen, z):
        """
         hist on specific train epoch, show generator input z, generated gen, data of real
        """
        plt.hist(real.reshape(-1), histtype='step', label='Real')
        plt.hist(gen.reshape(-1), histtype='step', label='Generated')
        plt.hist(z.reshape(-1), histtype='step', label='Input')
        plt.legend(loc=0)

    def print_stats(self, real, gen):
        """
        print real data and generated data
        """
        print('mu and std of Real:', (np.mean(real), np.std(real)))
        print('mu and std of Gen:', (np.mean(gen), np.std(gen)))


def main():
    # set hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu', type=float, default=4,
                        help='Mean of population data')
    parser.add_argument('--sigma', type=float, default=1.25,
                        help='Sigma of population data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for networks')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Epochs for the networks')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=200,
                        help='Input dimension for the generator.')
    parser.add_argument('--n_iter_D', type=int, default=1,
                        help='training iteration for D')
    parser.add_argument('--n_iter_G', type=int, default=5,
                        help='training iteration for G')
    args = parser.parse_args()

    # run model
    model = Model(args)
    model.fit()


if __name__ == '__main__':
    main()