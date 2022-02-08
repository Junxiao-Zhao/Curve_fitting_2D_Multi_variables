
"""
@author: Yongji Wang
@modified by: Junxiao Zhao
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pyDOE import lhs
import time
import math


class PINN:
    # Initialize the class
    def __init__(self, t_x, u, x_f, layers, lb, ub, gamma):   #自变量、因变量、范围内随机自变量、层数、最小边界、最大边界、loss_e权重

        self.t_x = t_x
        self.u = u
        self.x_f = x_f

        self.lb = lb
        self.ub = ub

        self.layers = layers

        self.gamma = gamma

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # Create a list including all training variables
        self.train_variables = self.weights + self.biases
        # Key point: anything updates in train_variables will be 
        #            automatically updated in the original tf.Variable

        # define the loss function
        self.loss = self.loss_NN()

        self.optimizer_Adam = tf.optimizers.Adam()

    '''
    Functions used to establish the initial neural network
    ===============================================================
    '''

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - [1.0, 1.0]
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    '''
    Functions used to building the physics-informed contrainst and loss
    ===============================================================
    '''
    def net_u(self, t):
        res = self.neural_net(t, self.weights, self.biases)
        return res
    
    def net_f(self, xf):
        t = xf[:, 0:1]
        x = xf[:, -1:]
        f = self.govern(t, x, self.net_u)
        return f

    def govern(self, t, x, func):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            tape2.watch(x)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(t)
                tape.watch(x)
                tx = tf.concat([t,x], 1)
                u = func(tx)
            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
        u_xx = tape2.gradient(u_x, x)
        f = u_t + u * u_x - tf.cast((0.01 / math.pi), dtype='float32') * u_xx
        return f


    @tf.function
    # calculate the physics-informed loss function
    def loss_NN(self):
        self.x_pred = self.net_u(self.t_x)
        loss_d = tf.reduce_mean(tf.square(self.u - self.x_pred))

        self.f_pred = self.net_f(self.x_f)
        loss_e = tf.reduce_mean(tf.square(self.f_pred))

        loss = loss_d + self.gamma * loss_e
        return loss#, loss_d, loss_e

    '''
    Functions used to define ADAM optimizers
    ===============================================================
    '''
    # define the function to apply the ADAM optimizer
    def Adam_optimizer(self, nIter):
        varlist = self.train_variables
        start_time = time.time()
        for it in range(nIter):
            tape = tf.GradientTape()
            self.optimizer_Adam.minimize(self.loss_NN, varlist, tape=tape)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.loss_NN()
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                #print(loss_d, loss_e)
                start_time = time.time()

    '''
    Function used for training the model
    ===============================================================
    '''
    def train(self, nIter):
        self.Adam_optimizer(nIter)


    def predict(self, t):
        u_p = self.net_u(t)
        return u_p


if __name__ == "__main__":
    noise = 0.0

    np.random.seed(123)
    tf.random.set_seed(123)

    N_tr = 200
    N_pd = 200
    layers = [2, 50, 50, 50, 50, 1]

    t = np.linspace(0, 1, 200, dtype='float32')[:, None]
    x = np.linspace(-1, 1, 200, dtype='float32')[:, None]
    T, X = np.meshgrid(t, x)

    tx0 = np.hstack((T[0:1, :].T, X[0:1, :].T))
    ux0 = np.zeros(x.shape, 'float32')
    
    tx1 = np.hstack((T[-1:, :].T, X[-1:, :].T))
    ux1 = np.zeros(x.shape, 'float32')
    
    t0x = np.hstack((T[:, 0:1], X[:, 0:1]))
    ut0 = -np.sin(math.pi * X[:, 0:1])
    
    X_u_train = np.vstack([tx0, tx1, t0x])
    u_train = np.vstack([ux0, ux1, ut0])

    lb_t = t.min(0)[0]
    ub_t = t.max(0)[0]
    lb_x = x.min(0)[0]
    ub_x = x.max(0)[0]

    X_f_train = [lb_t, lb_x] + [ub_t - lb_t , ub_x - lb_x] * lhs(2, 3200)
    X_f_train = np.vstack((X_f_train, X_u_train))

    X_u_train = tf.cast(X_u_train, dtype=tf.float32)    #边界条件
    u_train = tf.cast(u_train, dtype=tf.float32)  #边界条件对应值
    X_f_train = tf.cast(X_f_train,dtype=tf.float32) #随机点+边界点

    model = PINN(X_u_train, u_train, X_f_train, layers, tf.cast([lb_t, lb_x], dtype='float32'), tf.cast([ub_t, ub_x], dtype='float32'), 0.15)

    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    t_intp = np.linspace(0, 1, N_pd // 5, endpoint=False)
    x_intp = np.linspace(-1, 1, N_pd // 5, endpoint=False)
    T_intp, X_intp = np.meshgrid(t_intp[:, None], x_intp[:, None])
    
    intp = list()
    for i in range(N_pd // 5):
        for j in range(N_pd // 5):
            intp.append([T_intp[i][j], X_intp[i][j]])
    intp = tf.cast(intp, dtype=tf.float32)

    pred = tf.reshape(model.predict(intp), [40,40])

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(T_intp, X_intp, pred, cmap='rainbow')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.set_title("2D burgers' equation")

    sns.set()
    fig2 = plt.figure()
    
    sns.heatmap(pd.DataFrame(pred[::-1].numpy(), np.around(x_intp[::-1], 2), np.around(t_intp, 2)), cmap='rainbow')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('u(t,x)')

    fig3 = plt.figure()
    t_index = np.where(t_intp == 0.25)[0][0]
    plt.plot(x_intp, pred[:, t_index: t_index + 1].numpy())
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.title('t = 0.25')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    fig4 = plt.figure()
    t_index = np.where(t_intp == 0.50)[0][0]
    plt.plot(x_intp, pred[:, t_index: t_index + 1].numpy())
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.title('t = 0.50')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    fig5 = plt.figure()
    t_index = np.where(t_intp == 0.75)[0][0]
    plt.plot(x_intp, pred[:, t_index: t_index + 1].numpy())
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.title('t = 0.75')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.show()