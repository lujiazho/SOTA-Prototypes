#!/usr/bin/env python
# coding: utf-8

# Reference: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

# Two steps for training model on Multi GPUs on Multi nodes:
# 1. Initialize this process and join in a group with other processes
# 2. Set gpu number that we want to use for this process
# 3. Send each params that we want to compute with to GPU for this process
# 4. Wrap the model for data parallelism and model sync
# 5. Create data sampler for splitting data for each process with no overlap
# 6. Assign data sampler to DataLoader (Dataloader no need to shuffle because sampler does this by default)
# 
# Note:
# 1. For each node, we run following script with nr in config from 0 - N-1
# 2. Actual batch_size = batch_size per gpu (25) * world_size (4) = 100

# In[ ]:


import os
from datetime import datetime

import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def main(gpu, args):
    # differnet gpu has differnet rank
    rank = args.nr * args.g + gpu
    # step 1: Initialize this process and join up with the other processes
    # No process will continue until all processes have joined
    dist.init_process_group(                                   
        backend='nccl',              # NVIDIA Collective Communications Library (NCCL) as backend
        init_method='env://',        # tells the process group where to look for some settings
        world_size=args.world_size,                              
        rank=rank
    )
    
    torch.manual_seed(0)
    model = ConvNet()
    
    # step 2: set gpu number that we want to use
    torch.cuda.set_device(gpu)
    
    # step 3: send each params that we want to compute with to GPU
    model.cuda(gpu)
    
    # per gpu 25, so in total we actually run a batch_size of 25 * world_size
    batch_size = 25
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    # step 4: Wrap the model, ensuring data parallelism and backwards gradients are averaged
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    # step 5: makes sure that each process gets a different slice of the training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

class config:
    n = 1   # number of nodes
    g = 4   # number of gpus per node
    nr = 0  # the rank of the current node within all the nodes
    epochs = 1
    
    world_size = g*n  # total number of gpus == total number of processes
    master_addr = '10.7.3.62'
    master_port = '8888'

if __name__ == '__main__':
    args = config()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # start process for each gpu
    mp.spawn(main, nprocs=args.g, args=(args,))

