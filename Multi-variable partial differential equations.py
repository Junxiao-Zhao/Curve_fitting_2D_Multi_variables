
"""
@author: Yongji Wang
@modified by: Junxiao Zhao
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import time
import math


class PINN:
    # Initialize the class
    def __init__(self, t_x, u_v, x_f, layers, lb, ub, gamma):   #边界条件、因变量、范围内随机自变量、层数、最小边界、最大边界、loss_e权重

        self.t_x = t_x
        self.u_v = u_v
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
            if l < num_layers - 1:
                maximum = 1
            else:
                maximum = self.max
            W = self.xavier_init(size=[layers[l], layers[l + 1]], maximum = maximum)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size, maximum):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim)) * maximum
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y[:, 0:1], Y[:, -1:]

    '''
    Functions used to building the physics-informed contrainst and loss
    ===============================================================
    '''
    def net_u(self, t):
        u, v = self.neural_net(t, self.weights, self.biases)
        return u, v
    
    def net_f(self, t):
        f = self.govern(t[:, 0:1], t[:, -1:], self.net_u)
        return f

    def govern(self, x, y, func):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            tape2.watch(y)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                tape.watch(y)
                t = tf.concat([x,y], 1)
                u, v = func(t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            
        u_xx = tape2.gradient(u_x, x)
        u_xy = tape2.gradient(u_x, y)
        u_yy = tape2.gradient(u_y, y)

        v_xx = tape2.gradient(v_x, x)
        v_xy = tape2.gradient(v_x, y)
        v_yy = tape2.gradient(v_y, y)
        
        f = 4 * u_xx + 3 * v_xy + u_yy
        g = 4 * v_yy + 3 * u_xy + v_xx
        return f, g


    @tf.function
    # calculate the physics-informed loss function
    def loss_NN(self):
        u, v = self.net_u(self.t_x)
        loss_d = tf.reduce_mean(tf.square(self.u_v[:, 0:1] - u) + tf.square(self.u_v[:, -1:] - v))

        f, g = self.net_f(self.t_x)
        loss_e = tf.reduce_mean(tf.square(f) + tf.square(g))

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
        u_p, v_p = self.net_u(t)
        return tf.concat([u_p, v_p], 1)


if __name__ == "__main__":
    noise = 0.0

    np.random.seed(123)
    tf.random.set_seed(123)

    N_tr = 200
    N_pd = 200
    layers = [2, 50, 50, 50, 50, 2]

    x = np.linspace(-1, 1, 200, dtype='float32')[:, None]
    y = np.linspace(-1, 1, 200, dtype='float32')[:, None]
    X, Y = np.meshgrid(x, y)

    xy0 = np.hstack((X[0:1, :].T, Y[0:1, :].T))
    uy0 = vy0 = np.zeros(y.shape, 'float32')
    
    xy1 = np.hstack((X[-1:, :].T, Y[-1:, :].T))
    uy1 = vy1 = np.zeros(y.shape, 'float32')
    
    x0y = np.hstack((X[:, 0:1], Y[:, 0:1]))
    ux0 = -(1 - Y[:, 0:1] ** 2)
    vx0 = np.zeros(x.shape, 'float32')

    x1y = np.hstack((X[:, -1:], Y[:, -1:]))
    ux1 = 1 - Y[:, 0:1] ** 2
    vx1 = np.zeros(x.shape, 'float32')

    X_u_train = np.vstack([xy0, xy1, x0y, x1y])
    u_train = np.vstack([uy0, uy1, ux0, ux1])
    v_train = np.vstack([vy0, vy1, vx0, vx1])
    u_v_train = np.hstack([u_train, v_train])
    
    # Doman bounds
    lb_x = x.min(0)[0]
    ub_x = x.max(0)[0]
    lb_y = y.min(0)[0]
    ub_y = y.max(0)[0]

    X_f_train = [lb_x, lb_y] + [ub_x - lb_x, ub_y - lb_y] * lhs(2, 3200)
    X_f_train = np.vstack((X_f_train, X_u_train))

    X_u_train = tf.cast(X_u_train, dtype=tf.float32)    #边界条件
    u_v_train = tf.cast(u_v_train, dtype=tf.float32)  #边界条件对应值
    X_f_train = tf.cast(X_f_train,dtype=tf.float32) #随机点+边界点


    def U_fun_test(t):
        u = (1 - t[:, -1:] ** 2) * t[:, 0:1]
        return u
    
    def V_fun_test(t):
        return tf.cast(np.zeros([1600, 1]), dtype='float32')
    
    model = PINN(X_u_train, u_v_train, X_f_train, layers, tf.cast([lb_x, lb_y], dtype='float32'), tf.cast([ub_x, ub_y], dtype='float32'), 0.5)

    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    x_intp = np.linspace(-1, 1, N_pd // 5)[:, None]
    y_intp = np.linspace(-1, 1, N_pd // 5)[:, None]
    X_intp, Y_intp = np.meshgrid(x_intp, y_intp)
    
    intp = list()
    for i in range(N_pd // 5):
        for j in range(N_pd // 5):
            intp.append([X_intp[i][j], Y_intp[i][j]])
    intp = tf.cast(intp, dtype=tf.float32)

    T_u = tf.reshape(U_fun_test(intp), [40, 40])
    T_v = tf.reshape(V_fun_test(intp), [40, 40])
    pred = model.predict(intp)
    u_pred = tf.reshape(pred[:, 0:1], [40, 40])
    v_pred = tf.reshape(pred[:, -1:], [40, 40])
    

    error_u = np.linalg.norm(T_u - u_pred, 2) / np.linalg.norm(intp, 2)
    error_v = np.linalg.norm(T_v - v_pred, 2) / np.linalg.norm(intp, 2)
    print('Error u: %e, Error v: %e' % (error_u, error_v))

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig1 = plt.figure()
    ax_u_pred = Axes3D(fig1)
    ax_u_pred.plot_surface(X_intp, Y_intp, u_pred, cmap='rainbow')
    ax_u_pred.set_xlabel('x')
    ax_u_pred.set_ylabel('y')
    ax_u_pred.set_zlabel('u')
    ax_u_pred.set_title('u_pred')

    fig2 = plt.figure()
    T_ax_u = Axes3D(fig2)
    T_ax_u.plot_surface(X_intp, Y_intp, T_u, cmap='rainbow')
    T_ax_u.set_xlabel('x')
    T_ax_u.set_ylabel('y')
    T_ax_u.set_zlabel('u')
    T_ax_u.set_title('u_true_value')

    fig3 = plt.figure()
    ax_v_pred = Axes3D(fig3)
    ax_v_pred.plot_surface(X_intp, Y_intp, v_pred, cmap='rainbow')
    ax_v_pred.set_xlabel('x')
    ax_v_pred.set_ylabel('y')
    ax_v_pred.set_zlabel('v')
    ax_v_pred.set_title('v_pred')

    fig4 = plt.figure()
    T_ax_v = Axes3D(fig4)
    T_ax_v.plot_surface(X_intp, Y_intp, T_v, cmap='rainbow')
    T_ax_v.set_xlabel('x')
    T_ax_v.set_ylabel('y')
    T_ax_v.set_zlabel('v')
    T_ax_v.set_title('v_true_value')

    fig5 = plt.figure()
    error_u_ax = Axes3D(fig5)
    error_u_ax.plot_surface(X_intp, Y_intp, T_u - u_pred, cmap='rainbow')
    T_ax_v.set_xlabel('x')
    T_ax_v.set_ylabel('y')
    T_ax_v.set_zlabel('error')
    T_ax_v.set_title('error_u')

    fig6 = plt.figure()
    error_u_ax = Axes3D(fig6)
    error_u_ax.plot_surface(X_intp, Y_intp, T_v - v_pred, cmap='rainbow')
    T_ax_v.set_xlabel('x')
    T_ax_v.set_ylabel('y')
    T_ax_v.set_zlabel('error')
    T_ax_v.set_title('error_v')

    plt.show()
