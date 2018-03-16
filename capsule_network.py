#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

# library
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

# Hyper Parameters
USE_CUDA = True
BATCH_SIZE = 100
EPOCH = 30
LR = 1e-3
NUM_ROUTING_ITERATION = 3
DOWNLOAD_MNIST = False

if not(os.path.exists('./data')) or not os.listdir('./data'): # not mnist dir or mnist is empty dir
    DOWNLOAD_MNIST = True

# Mnist digits dataset
class Mnist:
    def __init__(self):
        train_data = tv.datasets.MNIST(
            root='./data',
            train=True,
            # transform to tensor (C x H x W) and normalize in the range [0.0, 1.0]
            # train_data.size (60000, 28, 28)
            # train_label.size (60000)
            transform=tv.transforms.ToTensor(),
            download=DOWNLOAD_MNIST,
        )

        test_data = tv.datasets.MNIST(
            root='./data',
            train=False,
            # test_data.size (10000, 28, 28)
            # test_label.size (10000)
            transform=tv.transforms.ToTensor(),
            download=DOWNLOAD_MNIST,
        )

        self.train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True, # ensure randomness in each epoch
        )
        
        self.test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

# The First Layer : Convolution Layer
## Input: 28*28*1
## Output: 20*20*256
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x): # x: (100, 1, 28, 28)
        x = self.conv(x) # Wn+1=(Wn+P*2−K)/S+1 x: (100, 256, 20, 20)
        output = F.relu(x) # output: (100, 256, 20, 20)
        return output

# The Second Layer :PrimaryCapsules Layer
# Input: 20*20*256
# Outout: 6*6*8*32
class PrimaryCaps(nn.Module):
    def __init__(self, in_channels=256, out_channels=32, kernel_size=9, stride=2, capsule_num=8):
        super(PrimaryCaps, self).__init__()

        # Use ModuleList to index capsules in PrimaryCaps
        self.capsules = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ) for _ in range(capsule_num)
        ])
        
    def forward(self, x):
        # x:(100, 256, 20, 20)
        # capsule(x): (100, 32, 6, 6)
        u = [capsule(x) for capsule in self.capsules] # u: [x x x x x x x x]
        u = torch.stack(u, dim=1) # u: (100, 8, 32, 6, 6)
        u = u.view(x.size(0), 32*6*6, -1) # u: (100, 32*6*6, 8)
        return self.squash(u)
    
    def squash(self, input_tensor):
        # input_tensor: (100, 32*6*6, 8)
        square_num = (input_tensor**2).sum(-1, keepdim=True) # square_num: (100, 32*6*6, 1)
        output_tensor = square_num * input_tensor / ((1 + square_num)*torch.sqrt(square_num)) # output_tensor: (100, 32*6*6, 8)
        return output_tensor

# The Third Layer: DigitCapsule Layer
# Input: 6*6*8*32
# Output: 16*10
class DigitCaps(nn.Module):
    def __init__(self, in_channels=8, out_channels=16, capsule_num=10, routes_num=32*6*6):
        super(DigitCaps, self).__init__()
        
        self.in_channels = in_channels
        self.routes_num = routes_num
        self.capsule_num = capsule_num

        self.W = nn.Parameter(torch.randn(1, routes_num, capsule_num, out_channels, in_channels)) # W: (1, 32*6*6, 10, 16, 8)

    def forward(self, x):
        # x: (100, 32*6*6, 8)
        x = torch.stack([x] * self.capsule_num, dim=2).unsqueeze(4) # x: (100, 32*6*6, 10, 8, 1)

        W = torch.cat([self.W] * BATCH_SIZE, dim=0) # W: (100, 32*6*6, 10, 16, 8)
        u_hat = torch.matmul(W, x) # u_hat: (100, 32*6*6, 10, 16, 1)

        b_ij = Variable(torch.zeros(1, self.routes_num, self.capsule_num, 1)) # b_ij: (1, 32*6*6, 10, 1)
        if USE_CUDA:
            b_ij = b_ij.cuda()

        # Dynamic routing
        num_iteration = 3
        for iteration in range(num_iteration):
            c_ij = F.softmax(b_ij) # c_ij: (1, 32*6*6, 10, 1)
            c_ij = torch.cat([c_ij] * BATCH_SIZE, dim=0).unsqueeze(4) # c_ij: (100, 32*6*6, 10, 1, 1)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # s_j: (100, 1, 10, 16, 1)
            v_j = self.squash(s_j) # v_j: (100, 1, 10, 16, 1)

            if iteration < num_iteration-1:
                # u_hatt.transpose(3, 4): (100, 32*6*6, 10, 1, 16)
                # torch.cat([v_j]*32*6*6, dim=1): (100, 32*6*6, 10, 16, 1)
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j]*self.routes_num, dim=1)) # a_ij: (100, 32*6*6, 10, 1, 1)
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True) # b_ij: (1, 32*6*6, 10, 1)

        return v_j.squeeze(1) # (100, 10, 16, 1)

    def squash(self, input_tensor):
        square_num = (input_tensor**2).sum(-1, keepdim=True)
        output_tensor = square_num * input_tensor / ((1 + square_num)*torch.sqrt(square_num))
        return output_tensor

# The Fourth Layer: Reconstruction
# Fully connected #1: Input:16*10 Output:512
# Fully connected #2: Input:512 Output: 1024
# Fully connected #3: Input:1024 Output:784
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(inplace=True), # 'inplace=true' can raise effciency
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
 

    def forward(self, x, data):
        # x: (100, 10, 16, 1)
        classes = torch.sqrt((x**2).sum(2)) # classes: (100, 10, 1)
        classes = F.softmax(classes) # classes: (100, 10, 1)

        _, max_length_index = classes.max(dim=1) # max_length_index: (100, 1)
        masked = Variable(torch.sparse.torch.eye(10)) # masked: (10, 10)
        if USE_CUDA:
            masked = masked.cuda()
        # masked contained the predict digit of each batch(one-hot vector)
        # index must be a vector instead of matrix
        masked = masked.index_select(dim=0, index=max_length_index.squeeze(1)) # masked: (100, 10)
        # x*masked[:, :, None, None]: (100, 10, 16, 1)
        # (masked[:, :, None, None]).view(100, -1): (100, 160)
        # input: highest capsule's output of each batch
        reconstructions = self.reconstruction_layers((x*masked[:, :, None, None]).view(x.size(0), -1)) # reconstruction (100, 784)
        reconstructions = reconstructions.view(-1, 1, 28, 28) # reconstruction (100, 1, 28, 28)
        
        return reconstructions, masked

# The capsule network
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        
        self.mse_loss = nn.MSELoss()

    def forward(self, x): # x: (100, 1, 28, 28)
        output=self.digit_capsules(self.primary_capsules(self.conv_layer(x))) # output: (100, 10, 16, 1)
        reconstruction, masked = self.decoder(output, data) # reconstruction: (100, 1, 28, 28), masked: (100, 10)
        return output, reconstruction, masked

    def loss(self, v_c, T_c, data, reconstructions):
        # v_c: (100, 10, 16, 1)
        # T_c: (100, 10)
        # data: (100, 1, 28, 28)
        # reconstruction: (100, 1, 28, 28)
        return self.margin_loss(v_c, T_c) + self.reconsturction_loss(data, reconstructions);

    def margin_loss(self, v_c, T_c):
        # v_c: (100, 10, 16, 1)
        # T_c: (100, 10)    
        _lambda = 0.5
        v_c_norm_sqrt = torch.sqrt((v_c**2).sum(2, keepdim=True)) # v_c_norm_sqrt: (100, 10, 1, 1)
        
        # ReLU function: relu(x) = max(0, x)
        left = (F.relu(0.9-v_c_norm_sqrt)**2).view(BATCH_SIZE, -1) # left: (100, 10)
        right = (F.relu(v_c_norm_sqrt-0.1)**2).view(BATCH_SIZE, -1) # right: (100, 10)

        loss = T_c*left + _lambda * (1.0-T_c) * right # loss: (100, 10)
        loss = loss.sum(dim=1).mean() # loss: (1), sum ->batch loss, that is sum of capsules' loss, mean-> meam of batch loss
        
        return loss

    def reconsturction_loss(self, data, reconstructions):
        # data: (100, 1, 28, 28)
        # reconstruction: (100, 1, 28, 28)
        # parm1:(100, 784), parm2:(100, 784)
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(data.size(0), -1)) # loss: (1)
        
        return 0.0005 * loss

# training and testing
capsule_net = CapsNet()
if USE_CUDA:
    capsule_net = capsule_net.cuda()
mnist = Mnist()
optimizer = torch.optim.Adam(capsule_net.parameters(), lr=LR)

for epoch in range(EPOCH):
    # training process
    capsule_net.train()
    for batch_id, (data, target) in enumerate(mnist.train_loader):
        
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
        
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(output, target, data, reconstructions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            print('train accuracy:', sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                        np.argmax(target.data.cpu().numpy(), 1)) / float(BATCH_SIZE))

    # testing process
    capsule_net.eval()
    for batch_id, (data, target) in enumerate(mnist.test_loader):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)

        if batch_id % 100 == 0:
            print('test accuracy:', sum(np.argmax(masked.data.cpu.numpy(), 1) ==
                        np.argmax(target.data.cpu.numpy(), 1)) / float(BATCH_SIZE))