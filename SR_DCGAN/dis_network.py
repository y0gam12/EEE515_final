import torch
import torch.nn as nn


n_in = 64
n = 64
n3 = 256
k = 3
s = 1
nrelu = 1
relu_init = 0.25
    
class SR_Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(SR_Discriminator, self).__init__()    
        self.ngpu = ngpu  
        self.relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv2d(kernel_size=k, out_channels= n, stride= s, in_channels=n_in, bias=False)
        self.bn1 = nn.BatchNorm2d(n)

        self.conv2 = nn.Conv2d(kernel_size=k, out_channels= n, stride= s*2, in_channels=n_in, bias=False)
        self.bn2 = nn.BatchNorm2d(n)

        self.conv3 = nn.Conv2d(kernel_size=k, out_channels= n*2, stride= s, in_channels=n_in, bias=False)
        self.bn3 = nn.BatchNorm2d(n*2)
        
        self.conv4 = nn.Conv2d(kernel_size=k, out_channels= n*2, stride= s*2, in_channels=n_in, bias=False)
        self.bn4 = nn.BatchNorm2d(n*2)

        self.conv5 = nn.Conv2d(kernel_size=k, out_channels= n*4, stride= s, in_channels=n_in, bias=False)
        self.bn5 = nn.BatchNorm2d(n*4)
        
        self.conv6 = nn.Conv2d(kernel_size=k, out_channels= n*4, stride= s*2, in_channels=n_in, bias=False)
        self.bn6 = nn.BatchNorm2d(n*4)

        self.conv7 = nn.Conv2d(kernel_size=k, out_channels= n*8, stride= s, in_channels=n_in, bias=False)
        self.bn7 = nn.BatchNorm2d(n*8)
        
        self.conv8 = nn.Conv2d(kernel_size=k, out_channels= n*8, stride= s*2, in_channels=n_in, bias=False)
        self.bn8 = nn.BatchNorm2d(n*8)

        self.d1024 = nn.Linear(512,2)
        self.d1 = nn.Linear(2,1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)

        x = self.d1024(x)
        x = self.relu(x)
        x = self.d1(x)

        x = nn.Sigmoid(x)

        return x















