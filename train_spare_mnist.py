#coding=utf-8
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy import *


#稀疏编码器代价计算
def sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
  
    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

  
    m = data.shape[1]

    # 正向传播
    a1 = data             
    z2 = W1.dot(a1) + b1.reshape((-1, 1))
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2.reshape((-1, 1))
    h  = sigmoid(z3)       
    y  = a1

   
    rho = sparsity_param
    rho_hat = np.mean(a2, axis=1)
    sparsity_delta = (-rho/rho_hat + (1.0-rho)/(1-rho_hat)).reshape((-1, 1))

    # 反向传播
    delta3 = (h-y)*sigmoid_prime(z3)
    delta2 = (W2.T.dot(delta3) + beta*sparsity_delta)*sigmoid_prime(z2)

    #代价计算
    squared_error_term = np.sum((h-y)**2) / (2.0*m)
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))
    cost = squared_error_term + weight_decay + sparsity_term

    #梯度计算
    W1grad = delta2.dot(a1.T)/m + lambda_*W1
    W2grad = delta3.dot(a2.T)/m + lambda_*W2
    b1grad = np.mean(delta2, axis=1)
    b2grad = np.mean(delta3, axis=1)
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad


#sigmod函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX)) 


#sigmod变种
def sigmoid_prime(x):
    f = sigmoid(x)
    df = f*(1.0-f)
    return df

def initialize_parameters(hidden_size, visible_size):

    # 随机初始化参数
    r  = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random((hidden_size, visible_size)) * 2.0 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2.0 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    theta = np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))

    return theta


def KL_divergence(p, q):
    
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))



#网络可视化
def display_network(A):
    opt_normalize = True
    opt_graycolor = True
    A = A - np.average(A)

    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    return image


def load_MNIST_images(filename):	#加载mnist图片
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        n_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((n_images, rows * cols))
        images = images.T
        images = images.astype(np.float64) / 255
        f.close()
        return images


def load_MNIST_labels(filename):	#mnist标签
    with open(filename, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        n_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        labels = np.fromfile(f, dtype=np.uint8)
        f.close()
        return labels




# 加载mnist数据集中的图片
images = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
patches = images[:, :10000]#选取前10000张

n_patches = patches.shape[1]    # 图片批次

# 随机展示20张图片并保存
image = display_network(patches[:, [np.random.randint(n_patches) for i in range(20)]])

plt.figure()
plt.imsave('sparse_autoencoder_minist_patches.png', image, cmap=plt.cm.gray)
plt.imshow(image, cmap=plt.cm.gray)

visible_size = patches.shape[0] # 输入单元数量
hidden_size = 196               # 隐藏单元数量

weight_decay_param = 3e-3       # 权重衰减系数　　学习率
beta = 3                        # 权重惩罚因子
sparsity_param = 0.1            # 隐藏单元平均激活数量

theta = initialize_parameters(hidden_size, visible_size)    #随机初始化参数

J = lambda theta : sparse_autoencoder_cost(theta, visible_size, hidden_size, weight_decay_param, sparsity_param, beta, patches)

# 迭代次数设置为500
options = {'maxiter': 500, 'disp': True, 'gtol': 1e-5, 'ftol': 2e-9}
results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)


#可视化权重单元
W1 = opt_theta[0:hidden_size*visible_size].reshape((hidden_size, visible_size))

image = display_network(W1.T)
plt.figure()
plt.imsave('sparse_autoencoder_minist_weights.png', image, cmap=plt.cm.gray)
plt.imshow(image, cmap=plt.cm.gray)

plt.show()


