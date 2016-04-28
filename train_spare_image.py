#coding=utf-8
import numpy as np
import scipy
from numpy import *
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.io import loadmat

#稀疏编码代价计算
def sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    m = data.shape[1]

　　　　#前向传播
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

    # 待机计算
    squared_error_term = np.sum((h-y)**2) / (2.0*m)
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))
    cost = squared_error_term + weight_decay + sparsity_term

    # 梯度计算
    W1grad = delta2.dot(a1.T)/m + lambda_*W1
    W2grad = delta3.dot(a2.T)/m + lambda_*W2
    b1grad = np.mean(delta2, axis=1)
    b2grad = np.mean(delta3, axis=1)
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad


#sigmod函数定义
def sigmoid(inX):
    return 1.0/(1+exp(-inX)) 


#sigmod函数变种
def sigmoid_prime(x):
    f = sigmoid(x)
    df = f*(1.0-f)
    return df


#初始化参数
def initialize_parameters(hidden_size, visible_size):
  
    # 随机初始化参数
    r  = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random((hidden_size, visible_size)) * 2.0 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2.0 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    theta = np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))

    return theta


#数值梯度检测
def check_numerical_gradient():
    x = np.array([4, 10], dtype=np.float64)
    value, grad = simple_quadratic_function(x)

    # Use your code to numerically compute the gradient of simple_quadratic_function at x.
    func = lambda x : simple_quadratic_function(x)[0] 
    numgrad = compute_numerical_gradient(func, x)

    n_grad = grad.size
    for i in range(n_grad):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12 
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff) 
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')


def KL_divergence(p, q):
    
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


#二次函数
def simple_quadratic_function(x):
 
    value = x[0]*x[0] + 3*x[0]*x[1]

    grad = np.zeros(2)
    grad[0]  = 2*x[0] + 3*x[1]
    grad[1]  = 3*x[0]

    return value, grad

def compute_numerical_gradient(J, theta):　#数值梯度计算
 
    n = theta.size
    grad = np.zeros(n)
    eps = 1.0e-4
    eps2 = 2*eps
    
    for i in range(n):
        theta_p = theta.copy()
        theta_n = theta.copy()
        theta_p[i] = theta[i] + eps
        theta_n[i] = theta[i] - eps
        
        grad[i] = (J(theta_p) - J(theta_n)) / eps2
    


#网络显示
def display_network(A):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
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


#取样数据图片
def sample_images(fname):

    image_data = loadmat(fname)['IMAGES']
    image_rows = image_data.shape[0]
    image_cols = image_data.shape[1]
    n_images = image_data.shape[2]

    n_patches = 10000
    patch_rows = 8
    patch_cols = 8
    patches = np.zeros((patch_rows*patch_cols, n_patches), dtype=np.float64)

    rows_diff = image_rows - patch_rows
    cols_diff = image_cols - patch_cols
    for i in range(n_patches):
        image_id = np.random.randint(0, n_images)
        x = np.random.randint(0, rows_diff)
        y = np.random.randint(0, cols_diff)
        patch = image_data[y:y+patch_rows, x:x+patch_cols, image_id].ravel()
        patches[:, i] = patch

    # Normalize data
    patches = normalize_data(patches)

    return patches


#归一化图片数据
def normalize_data(patches):

    # Remove the DC (mean of images)
    patches -= patches.mean(axis=0)

    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3.0 * np.std(patches)
    patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches + 1) * 0.4 + 0.1

    return patches

"""
第一步：加载图片数据，显示20张图片    
"""
patches = sample_images('data/IMAGES.mat') # 数据加载

n_patches = patches.shape[1] #获取数据批次

# 随机200张图显示并保存
image = display_network(patches[: , [np.random.randint(n_patches) for i in range(20)]])

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

plt.imsave('sparse_autoencoder_patches.png', image, cmap=plt.cm.gray)


"""
第二步：设置spare autoencode的滤波器参数，经验值，不需要改变
"""
visible_size = patches.shape[0] # 输入单元数目
hidden_size = 25                # 隐藏单元数目

weight_decay_param = 0.0001 # 权重衰减系数　学习率
beta = 3                    # 权重惩罚因子
sparsity_param = 0.01       # 隐藏单元平均激活参数

#  随机初始化参数
theta = initialize_parameters(hidden_size, visible_size)

"""
第三步：计算代价
"""

cost, grad = sparse_autoencoder_cost(theta, visible_size, hidden_size, weight_decay_param, sparsity_param, beta, patches)

"""
第四步：梯度校验
"""
debug = False # debug设置为true时进行梯度校验　　

if debug:
    check_numerical_gradient()

    J = lambda theta : sparse_autoencoder_cost(theta, visible_size, hidden_size,
        weight_decay_param, sparsity_param, beta, patches)[0]
    numgrad = compute_numerical_gradient(J, theta)

    # Use this to visually compare the gradients side by side
    n = min(grad.size, 20) # Number of gradients to display
    for i in range(n):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Compare numerically computed gradients with the ones obtained from backpropagation
    # This should be small. In our implementation, these values are usually less than 1e-9.
    # When you got this working, Congratulations!!!
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff)


"""
随机初始化参数
"""
theta = initialize_parameters(hidden_size, visible_size)

J = lambda theta : sparse_autoencoder_cost(theta, visible_size, hidden_size,
    weight_decay_param, sparsity_param, beta, patches)


options = {'maxiter': 400, 'disp': True, 'gtol': 1e-5, 'ftol': 2e-9}
results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)


"""
第六步：可视化
"""
W1 = opt_theta[0:hidden_size*visible_size].reshape((hidden_size, visible_size))

print("Save and show the W1")
image = display_network(W1.T)

plt.figure()
plt.imsave('sparse_autoencoder_weights.png', image, cmap=plt.cm.gray)
plt.imshow(image, cmap=plt.cm.gray)

plt.show()


