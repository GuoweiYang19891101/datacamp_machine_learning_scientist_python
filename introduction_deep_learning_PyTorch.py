# Introduction to Deep Learning with PyTorch

# setting working directory
import os

os.chdir('E:/DataCampProjects/Deep Learning')

# CHAPTER 1. Introduction to PyTorch

# Lesson 1.1 Introduction to PyTorch

# there are tons of deep learning libraries out there
# we choose PyTorch because
# 1) easy to use
# 2) strong GPU support, models run fast
# 3) many algorithms are already implemented
# 4) automatic differentiation, more in text lesson
# 5) similar to NumPy

# EXERCISE:
# 1) creating tensors in PyTorch
# import torch
import torch

# create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# calculate the shape of the tensor
tensor_size = your_first_tensor.shape

# 2) matrix multiplication
# create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)

# do an element-wise multiplication of tensor_of_ones and identity_tensor
element_multiplication = tensor_of_ones * identity_tensor

# Lesson 1.2 Forward propagation

# use calculation operations to calculate hidden layers, all the way to output layer

# EXAMPLE:
# 1) Forward pass
# initialize tensors x, y, and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# multiply x with y
q = torch.matmul(x, y)

# multiply element-wise z with 1
f = z * q

mean_f = torch.mean(f)
print(mean_f)

# Lesson 1.3 Backpropagation by auto-differentiation

# basic alrogithm in nerual network: backpropagation
# derivatives are one of the central concepts in calculus
# it can be used to describe the steepness
# derivative rules:
# (f+g)' = f' + g'
# (f.g)' = f.dg + g.df
# (x^n)' = (d/dx)x^n = nx^(n-1)
# (1/x)' = -(1/x^2)
# (f/g)' = (df.1/g) + ((-1/g^2)dg.f)

# EXERCISE:
# 1) backpropagation using PyTorch
# initialize x, y and z values
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# set q and f
q = x + y
f = q * z

# compute the derivatives
f.backward()

# print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))

# 2) calculating gradients in PyTorch
# initialize tensors x, y, and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# multiply tensor x and y
q = torch.matmul(x, y)

# elementwise multiply tensors z with q
f = z * q

mean_f = torch.mean(f)

# calculate the gradients
mean_f.backward()

# Lesson 1.4 Introduction to Neural Networks

# there are a lot of good classifiers when data is given on vectorial format as features
# when data is not given as features, like images, speech, text or video, neural networks are better
# PyTorch has its style to build a neural network, which is object-oriented

# EXERCISE:
# 1) Your first nerual network
# initialize the weights of the neural network
input_layer = torch.tensor([0.4391, 0.2166, 0.0727, 0.6066, 0.2018, 0.5647, 0.8097, 0.7429, 0.8383,
                            0.4565, 0.1714, 0.2783, 0.8526, 0.9207, 0.8141, 0.2218, 0.1059, 0.4776,
                            0.0876, 0.7910, 0.7050, 0.9185, 0.5234, 0.6919, 0.8116, 0.9154, 0.5392,
                            0.1467, 0.7782, 0.7417, 0.2321, 0.3900, 0.6797, 0.5984, 0.4987, 0.3920,
                            0.6738, 0.3391, 0.0899, 0.0434, 0.8340, 0.8373, 0.4745, 0.5674, 0.7439,
                            0.8614, 0.3817, 0.8949, 0.3824, 0.4512, 0.6160, 0.7078, 0.8680, 0.4301,
                            0.4867, 0.8855, 0.8867, 0.5113, 0.3762, 0.6612, 0.6501, 0.0276, 0.7206,
                            0.6033, 0.0579, 0.9036, 0.1049, 0.4639, 0.8371, 0.4950, 0.6365, 0.5715,
                            0.3712, 0.4051, 0.6928, 0.5901, 0.3047, 0.7129, 0.0925, 0.0450, 0.2836,
                            0.6397, 0.0308, 0.4298, 0.2981, 0.2196, 0.1764, 0.2655, 0.6156, 0.5357,
                            0.6822, 0.9702, 0.5570, 0.4491, 0.1041, 0.7619, 0.6958, 0.1115, 0.8647,
                            0.6945, 0.2473, 0.7779, 0.2479, 0.5541, 0.2568, 0.1435, 0.1220, 0.9974,
                            0.8812, 0.5378, 0.0138, 0.4780, 0.5430, 0.1642, 0.6709, 0.5163, 0.1254,
                            0.3469, 0.1703, 0.9396, 0.9143, 0.7492, 0.3613, 0.0574, 0.2188, 0.4316,
                            0.3838, 0.8311, 0.9308, 0.3411, 0.2480, 0.9818, 0.9010, 0.0820, 0.4227,
                            0.8861, 0.2897, 0.9272, 0.5562, 0.9484, 0.9851, 0.8297, 0.5763, 0.9014,
                            0.0967, 0.0776, 0.1327, 0.3345, 0.2796, 0.3958, 0.9325, 0.3417, 0.8131,
                            0.7632, 0.0274, 0.9376, 0.1053, 0.7940, 0.2158, 0.8740, 0.0217, 0.3962,
                            0.2603, 0.3652, 0.7623, 0.3015, 0.4989, 0.6998, 0.5023, 0.5157, 0.3113,
                            0.7357, 0.9911, 0.3227, 0.4917, 0.6058, 0.0023, 0.8144, 0.8023, 0.9329,
                            0.4008, 0.5434, 0.2761, 0.0796, 0.8602, 0.1491, 0.5056, 0.7638, 0.8327,
                            0.0452, 0.9965, 0.1765, 0.9519, 0.8541, 0.2157, 0.3961, 0.8792, 0.1069,
                            0.0743, 0.0937, 0.1853, 0.9177, 0.1962, 0.9983, 0.2855, 0.0416, 0.8543,
                            0.4501, 0.6212, 0.2389, 0.9135, 0.3791, 0.7217, 0.0501, 0.5466, 0.8225,
                            0.0544, 0.8254, 0.7833, 0.1644, 0.6821, 0.8258, 0.9084, 0.1075, 0.6621,
                            0.5790, 0.3401, 0.8574, 0.7662, 0.3414, 0.2374, 0.5633, 0.1223, 0.3969,
                            0.1307, 0.6264, 0.4733, 0.4801, 0.6976, 0.3488, 0.5693, 0.2913, 0.3497,
                            0.1289, 0.4521, 0.1727, 0.5682, 0.6562, 0.8367, 0.7101, 0.8584, 0.4579,
                            0.4979, 0.4587, 0.9905, 0.9619, 0.1044, 0.9920, 0.5680, 0.3565, 0.1031,
                            0.1264, 0.0551, 0.7876, 0.9197, 0.9014, 0.6118, 0.2766, 0.3549, 0.3078,
                            0.6476, 0.8701, 0.4710, 0.4979, 0.3664, 0.8644, 0.7643, 0.0329, 0.7594,
                            0.6311, 0.8371, 0.5769, 0.4869, 0.3889, 0.0191, 0.3808, 0.3211, 0.0632,
                            0.3596, 0.8431, 0.5618, 0.0913, 0.1305, 0.5886, 0.2554, 0.7681, 0.6039,
                            0.9326, 0.1824, 0.8807, 0.1565, 0.5949, 0.2285, 0.1116, 0.3089, 0.4196,
                            0.2382, 0.5711, 0.5023, 0.2986, 0.4160, 0.2273, 0.6092, 0.2880, 0.3244,
                            0.8952, 0.6831, 0.2838, 0.9062, 0.9557, 0.7465, 0.8997, 0.1730, 0.6998,
                            0.6885, 0.6107, 0.9739, 0.9127, 0.2716, 0.5652, 0.4718, 0.6818, 0.3631,
                            0.5693, 0.3986, 0.9977, 0.0676, 0.0906, 0.2865, 0.4730, 0.2544, 0.3758,
                            0.8158, 0.2085, 0.6046, 0.0254, 0.0701, 0.0084, 0.0659, 0.8882, 0.0017,
                            0.7615, 0.2124, 0.9906, 0.5704, 0.5821, 0.1531, 0.6816, 0.7272, 0.8515,
                            0.9539, 0.1227, 0.6820, 0.4794, 0.1531, 0.7027, 0.0674, 0.4053, 0.1106,
                            0.6508, 0.3728, 0.8017, 0.4771, 0.0629, 0.7894, 0.7566, 0.5623, 0.2281,
                            0.1028, 0.5438, 0.5711, 0.6695, 0.9871, 0.2686, 0.4283, 0.4036, 0.9038,
                            0.6575, 0.7247, 0.8284, 0.5737, 0.1689, 0.2920, 0.9170, 0.2577, 0.3078,
                            0.5105, 0.6608, 0.4127, 0.9216, 0.1905, 0.8795, 0.0295, 0.4196, 0.9025,
                            0.0551, 0.2541, 0.2610, 0.4855, 0.9716, 0.1299, 0.1980, 0.0988, 0.2448,
                            0.1412, 0.2173, 0.6206, 0.5575, 0.5135, 0.7787, 0.8362, 0.3398, 0.6100,
                            0.1126, 0.0445, 0.9104, 0.7605, 0.7098, 0.1871, 0.4933, 0.9394, 0.1857,
                            0.3401, 0.2387, 0.2774, 0.5337, 0.7424, 0.7993, 0.2910, 0.4165, 0.2029,
                            0.4070, 0.8290, 0.8284, 0.8405, 0.9084, 0.0323, 0.5652, 0.5351, 0.2467,
                            0.3284, 0.2465, 0.5818, 0.9819, 0.9321, 0.5981, 0.3996, 0.3471, 0.8582,
                            0.0935, 0.3030, 0.6923, 0.2868, 0.9089, 0.6269, 0.5245, 0.4735, 0.3754,
                            0.2239, 0.9166, 0.8479, 0.2107, 0.1926, 0.6596, 0.8620, 0.0363, 0.5945,
                            0.8342, 0.3558, 0.3522, 0.2343, 0.3839, 0.8219, 0.6693, 0.3993, 0.8799,
                            0.4648, 0.7028, 0.9005, 0.0081, 0.0543, 0.6507, 0.9398, 0.2393, 0.8488,
                            0.3269, 0.2925, 0.4563, 0.8795, 0.0078, 0.2617, 0.6105, 0.1521, 0.5920,
                            0.7176, 0.2059, 0.2639, 0.4145, 0.3840, 0.4021, 0.9518, 0.4168, 0.9400,
                            0.1801, 0.7388, 0.2773, 0.8421, 0.8376, 0.0736, 0.7202, 0.5488, 0.4185,
                            0.5385, 0.9563, 0.0986, 0.5444, 0.0170, 0.1961, 0.3652, 0.6194, 0.3102,
                            0.4419, 0.5340, 0.7912, 0.0160, 0.5004, 0.3980, 0.7651, 0.1616, 0.6263,
                            0.9798, 0.4603, 0.0044, 0.7390, 0.6294, 0.9898, 0.5763, 0.4262, 0.7998,
                            0.6838, 0.8906, 0.7005, 0.2135, 0.3512, 0.0197, 0.0342, 0.4098, 0.8963,
                            0.8598, 0.1440, 0.7754, 0.7866, 0.9654, 0.0862, 0.8344, 0.3979, 0.5522,
                            0.6647, 0.0865, 0.2102, 0.8714, 0.0425, 0.2722, 0.9267, 0.6379, 0.4873,
                            0.8371, 0.5212, 0.1434, 0.1324, 0.5694, 0.5003, 0.0954, 0.5175, 0.3906,
                            0.1730, 0.9177, 0.4648, 0.9995, 0.8235, 0.9038, 0.1416, 0.7794, 0.9023,
                            0.0468, 0.1387, 0.8888, 0.2877, 0.3881, 0.3172, 0.7519, 0.3286, 0.6424,
                            0.4991, 0.1537, 0.6861, 0.9796, 0.3028, 0.2135, 0.6573, 0.6289, 0.8975,
                            0.5694, 0.3031, 0.3009, 0.4799, 0.0524, 0.7395, 0.9347, 0.5061, 0.7431,
                            0.3330, 0.7043, 0.1093, 0.8722, 0.1318, 0.9557, 0.1774, 0.8075, 0.7857,
                            0.9795, 0.2327, 0.9950, 0.2880, 0.9842, 0.2894, 0.7889, 0.3470, 0.6801,
                            0.5327, 0.9203, 0.3631, 0.5272, 0.3374, 0.1745, 0.3565, 0.8040, 0.8187,
                            0.6512, 0.2833, 0.7385, 0.4610, 0.7315, 0.5133, 0.1566, 0.4231, 0.6987,
                            0.9548, 0.5575, 0.2874, 0.6357, 0.0939, 0.7951, 0.9864, 0.0211, 0.1178,
                            0.3391, 0.7828, 0.1564, 0.1516, 0.9391, 0.9591, 0.1546, 0.3951, 0.9380,
                            0.1317, 0.0023, 0.1260, 0.6995, 0.3098, 0.4642, 0.5679, 0.8507, 0.0320,
                            0.4200, 0.2349, 0.9652, 0.9342, 0.5645, 0.0592, 0.8056, 0.2017, 0.1913,
                            0.7302, 0.1536, 0.8517, 0.4204, 0.4626, 0.3712, 0.5769, 0.8889, 0.8145,
                            0.4001, 0.0646, 0.3542, 0.1200, 0.9486, 0.3305, 0.6670, 0.8237, 0.8114,
                            0.6054, 0.3776, 0.7591, 0.5693, 0.1200, 0.3625, 0.9003, 0.9227, 0.0346,
                            0.1366, 0.4425, 0.7668, 0.6029, 0.4632, 0.9410, 0.9560, 0.0418, 0.0384,
                            0.1423, 0.3460, 0.2600, 0.8934, 0.6335, 0.3471, 0.0428, 0.7297, 0.5015,
                            0.2242, 0.9668, 0.2681, 0.8323, 0.0690, 0.6483, 0.2807, 0.5231, 0.1563,
                            0.1020, 0.9856, 0.0609, 0.7152, 0.0291, 0.2732, 0.2617, 0.3956, 0.3214,
                            0.3451, 0.9367, 0.5870, 0.7651, 0.2072, 0.0748, 0.9630, 0.9924, 0.4729,
                            0.4122, 0.8208, 0.3499, 0.1131, 0.8607, 0.9053, 0.2457, 0.4597, 0.5308,
                            0.7512, 0.5754, 0.4322, 0.6558, 0.8472, 0.6351, 0.5246, 0.9171, 0.2380,
                            0.2300])
weight_1 = torch.rand(784, 200)
weight_2 = torch.rand(200, 10)

# multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# multiply hidden_1 and weight_2
output_layer = torch.matmul(hidden_1, weight_2)
print(output_layer)

# 2) Your first PyTorch neural network
import torch
import torch.nn as nn


class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()

        # instantiate all 2 linear layers
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# CHAPTER 2. Artificial Neural Networks


# Lesson 2.1 Activation functions

# matrix multiplication is just a linear transformation, we can simplify any neural network in a
# single layer neural network
# our neural nets can only separate linearly separable datasets
# activation functions are non-linear functions inserted in each layer of the neural network
# activation functions:
# sigmoid, leaky ReLu, tanh, maxout, ReLu, ELU
# the most common activation function is ReLu

# EXERCISE:
# 1) Neural networks
# initialize input_layer and weights
import torch

input_layer = torch.tensor([[0.0401, -0.9005, 0.0397, -0.0876]])
weight_1 = torch.tensor([[-0.1094, -0.8285, 0.0416, -1.1222],
                         [0.3327, -0.0461, 1.4473, -0.8070],
                         [0.0681, -0.7058, -1.8017, 0.5857],
                         [0.8764, 0.9618, -0.4505, 0.2888]])
weight_2 = torch.tensor([[0.6856, -1.7650, 1.6375, -1.5759],
                         [-0.1092, -0.1620, 0.1951, -0.1169],
                         [-0.5120, 1.1997, 0.8483, -0.2476],
                         [-0.3369, 0.5617, -0.6658, 0.2221]])
weight_3 = torch.tensor([[0.8824, 0.1268, 1.1951, 1.3061],
                         [-0.8753, -0.3277, -0.1454, -0.0167],
                         [0.3582, 0.3254, -1.8509, -1.4205],
                         [0.3786, 0.5999, -0.5665, -0.3975]])

# calculate the first and second hidden layer
hidden_1 = torch.matmul(input, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)

# calculate the output
print(torch.matmul(hidden_2, weight_3))

# calculate weight_composed_1 and weight
weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)

# multiply input_layer and weight
print(torch.matmul(input_layer, weight))

# 2) ReLU activation
# instantiate non-linearity
relu = nn.ReLU()

# apply non-linearity on the hidden layers
hidden_1_activated = relu(torch.matmul(input_layer, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
print(torch.matmul(hidden_2_activated, weight_3))

# apply non-linearity in the product of first two weights
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))

# multiply 'weight_composed_1_activated' with 'weight_3'
weight = torch.matmul(weight_composed_1_activated, weight_3)

# multiply input_layer with weight
print(torch.matmul(input_layer, weight))

# Lesson 2.2 Loss functions

# initialize neural networks with random weights
# do a forward pass
# calculate a loss function (1 number)
# calculate the gradients
# change the weights based on gradients
# for regression: least squared loss
# for classification: softmax cross-entropy loss
# for more complicated problems, we need more complicated loss
# important thing is that losses should be differentiable,
# otherwise we wouldn't be able to compute gradients
# softmax is a function which transforms nubmers into probabilities

# EXERCISE:
# 1) calculating loss function in PyTorch
# initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])

# instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()

# compute and print the loss
loss = criterion(logits, ground_truth)
print(loss)

# 2) loss function of random scores
# import torch and torch.nn
import torch
import torch.nn as nn

# initialize logits and ground truth
logits = torch.rand(1, 1000)
ground_truth = torch.tensor([111])

# instantiate cross-entropy loss
criterion = nn.CrossEntropyLoss()

# calculate and print the loss
loss = criterion(logits, ground_truth)
print(loss)

# Lesson 2.3 Preparing a dataset in PyTorch

# in order to use datasets in PyTorch, they need to be converted to a PyTorch friendly format

# EXERCISE:
# 1) preparing MNIST dataset
# import modules
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

# transform the data to torch tensors and normalize it
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307), (0.3081))])

# prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, download=True,
                                      transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False, download=True,
                                     transform=transform)

# prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)

# 2) inspecting the dataloaders
# compute the shape of training and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# compute the size of minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Lesson 2.4 Training neural networks

# combine previous lessons to form this recipe to train neural networks:
# recipe for training neural networks
# prepare the dataloaders (2.3)
# build a neural network (1.4 & 2.1)

# Loop over following:
# do a forward pass (1.2)
# calculate loss function (2.2)
# calculate the gradients (1.3)
# change the weights based on gradients (weight = weight - gradient * learning_rate)

# EXERCISE:
# 1) building a neural network - again

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functions as F
import torch.optim as optim

# load the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307), (0.3081))])

# prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, download=True,
                                      transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False, download=True,
                                     transform=transform)

# prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)


# define the class Net
class Net(nn.Module):
    def __init__(self):
        # define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 2) training a neural network
# instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()
optimizer = optim.Adam(model.parameters(), lr=3e-4) # specify optimizer and learning rate
criterion = nn.CrossEntropyLoss() # specify the loss function

for batch_idx, data_target in enumerate(trainloader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # complete a forward pass
    output = model(data)

    # compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 3) using the network to make predictions
# initiate total and correct
correct, total = 0, 0
predictions = []

# set the model in eval(test) mode
model.eval()

for i, data in enumerate(testloader, 0):
    inputs, labels = data

    # put each image into a vector
    inputs = inputs.view(-1, 28 * 28)

    # do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()

print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

# CHAPTER 3. Convolutional Neural Networks (CNNs)


# Lesson 3.1 Convolution operator

# there are problems with fully-connected neural networks
# we do not need to consider all relationships
# they are not computationally efficient
# too many parameters, easy to over-fit
# main ideas:
# units are connected with only a few units from the previous layer
# units share weights
# these ideas lead to convolutional neural networks (CNNs)

# convolution operator:
# a convolution operator is to use a filter to convolve through the image
# then, convolution operators form an activation map/ feature

# two ways of convolutions in PyTorch:
# 1) OOP-based/ object-oriented way (torch.nn)
# 2) functional (torch.nn.functional)

# EXERCISE:
# 1) convolution operator - OOP way
import torch
import torch.nn as nn

# create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)

# convolve the image with the filters
output_feature = conv_filters(images)
print(output_feature.shape)

# 2) convolution operator - Functional way
import torch
import torch.nn.functional as F

# create 10 random images
image = torch.rand(10, 1, 28, 28)

# create 6 filters
filters = torch.rand(6, 1, 3, 3)

# convolve the image with the filters
output_features = F.conv2d(image, filter, stride=1, padding=1)
print(output_features.shape)

# Lesson 3.2 Pooling operator

# convolutions are used to extract features from the image
# pooling is a way of feature selection
# two most important operators are max-pooling and average-pooling
    # 1) max-pooling takes the maximum number in region of images
    # 2) average-pooling takes the average value in region of images

# two ways of pooling in PyTorch:
# 1) OOP-based/ object-oriented way (torch.nn)
# 2) functional (torch.nn.functional as F)

# EXERCISE:
# 1) max-pooling operator
import torch
import torch.nn
import torch.nn.functional as F

im = torch.Tensor([[[[3, 1, 3, 5], [6, 0, 7, 9], [3, 2, 1, 4], [0, 2, 4, 3]]]])

# build a pooling operator with size '2'
max_pooling = torch.nn.MaxPool2d(2)

# apply the pooling operator
output_feature = max_pooling(im)

# use pooling operator in the image
output_feature_F = F.max_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

# 2) average-pooling operator
# build a pooling operator with size '2'
avg_pooling = torch.nn.AvgPool2d(2)

# apply the pooling operator
output_feature = avg_pooling(im)

# use pooling operator in the image
output_feature_F = F.avg_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

# Lesson 3.3 Convolutional Neural Networks

# computer visions all have CNNs play a part in it
# AlexNet is the most influential CNN model that published

# EXERCISE:
# 1) your first CNN-__init__ method
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)

        # instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    # 2) your first CNN-forward() method

    def forward(self, x):
        # apply conv followed by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # apply conv followed by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # apply the fully connected layer and return the result
        return self.fc(x)


# Lesson 3.4 Training Convolutional Neural Networks

# there is not much difference between training of fully-connected and convolutional neural networks

# EXERCISE:
# 1) training CNNs
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 2) using CNNs to make predictions
# iterate over the data in the testloader
for i, data in enumerate(testloader):
    # get the image and label from data
    input, label = data

    # make a forward pass in the net with your image
    output = net(input)

    # argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction" + str(predicted))
    else:
        print("Your net prediction was" + str(predicted) + ", but the correct label is: " + str(label))


# CHAPTER 4. Using Convolutional Neural Networks


# Lesson 4.1 The sequential module

# sequential modules can help us to create neural networks with much less code

# EXERCISE:
# 1) sequential module - init method


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # declare all the layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True)
        )

        # declare all layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 40, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 10)
        )

    # 2) sequential module - forward() method

    def forward(self, x):
        # apply the feature extractor in the input
        x = self.features(x)

        # squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)

        # classify the images
        x = self.classifier(x)
        return x

# Lesson 4.2 The problem of over-fitting

# the biggest problem in deep learning and machine learning is over-fitting
# over-fitting happens when there is a big difference between training and testing accuracy, also called high variance
# we introduce the validation set, also known as cross validation to avoid using testing set too many times
    # training set: train the model
    # validation set: select the model
    # testing set: test the model
# NOTE: it is important that the testing set is used only once, or the result is not trustworthy
# Also, the training set and validation set DO NOT overlap each other

# EXERCISE:
# 1) validation set
# shuffle the indices
import numpy as np
indices = np.arange(60000)
np.random.shuffle(indices)

# build the train loader
import torch
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                                                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))

# build the validation loader
val_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                         batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:]))


# Lesson 4.3 Regularization techniques

# introduce a few methods to train models more efficiently
# 1) L2-regularization
# L2-regularization is a method used in regression or SVM
# to apply it, we just add second term in loss function by setting 'weight_decay'

# 2) Dropout
# during each forward pass, there is a probability for each unit to be dropped from the computation
# it is used in fully-connected layers, rarely used in convolutional layers
# nn.Dropout(p=0.5) to set the probability to 0.5

# 3) Batch-normalization
# this is an important technique used nowadays in practically every neural network
# it computes the mean and variance of the mini-batch for each feature
# then it normalizes the features based on stats
# it is great on large neural networks
# nn.BatchNorm2d() will do the work

# 4) early-stopping
# it checks the accuracy of the network in the validation set at the end of each epoch
# after n epochs the performance hasn't increased, then training is terminated

# these techniques behave differently on training and evaluation mode
# it is important to set the net in the correct mode, or the training and evaluation will be broken

# EXERCISE:
# 1) L2-regularization

# instantiate the network
model = Net()

# instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()

# instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

# 2) dropout


class Net(nn.Module):
    def __init__(self):

        # define all the parameters of the net
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10)
        )

    def forward(self, x):

        # do the forward pass
        return self.classifier(x)


# 3) batch-normalization


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # implement the sequential module for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.BathNorm2d(10), # set the number to the number of channels
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.BathNorm2d(20) # set the number to the number of channels
        )

        # implement the fully connected layer for classification
        self.fc = nn.Linear(in_features=20*7*7, out_features=10)



# Lesson 4.4 Transfer learning

# in CNN network, the deeper you go, the more abstract the features become
# we use fine-tuning technique to be able to train CNN in small datasets
# there are two ways of fine-tuning neural networks:
    # 1) freeze most of layers (not updating them during back-propagation) and fine-tuning only the last few layers
    # 2) fine-tuning everything

# EXERCISE:
# 1) fine-tuning a CNN
# create a new model
model = Net()

# Load the parameters from the old model
model.load_state_dict(torch.load('my_net.pth'))

# change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()) )

# 2) torchvision module
# import the module
import torchvision

# download resnet18
model = torchvision.models.resnet18(pretrained=True)

# freeze all the layers bar the last one
for param in model.parameters():
    param.requires_grad=False

# change the number of output units
model.fc = nn.Linear(512, 7)