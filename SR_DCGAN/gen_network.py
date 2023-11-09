import torch
import torch.nn as nn




n_in = 64
n = 64
n3 = 256
k = 3
s = 1
nrelu = 1
relu_init = 0.25

class genblock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(genblock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x,identity)

        return x
    
class SR_Generator(nn.Module):
    def __init__(self, ngpu):
        super(SR_Generator, self).__init__()    
        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(in_channels = n_in, out_channels=n, kernel_size=k, stride=s)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU(num_parameters=nrelu, init=relu_init)
        self.conv2 = nn.Conv2d(in_channels=n_in, out_channels=n, kernel_size=k*3, stride=s)
        self.conv3 = nn.Conv2d(in_channels=n_in, out_channels=n*4, kernel_size=k, stride=s)
        self.shuff = nn.PixelShuffle(2)
        self.conv4 = nn.Conv2d(in_channels=n*4, out_channels=3, kernel_size=k,stride=s)
        

    def forward(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        identity = x
        identity_fst = x

        # Residual Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x, identity)
        # End of 20 residual Blocks

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.add(x,identity_fst)

        x = self.conv3(x)
        x = self.shuff(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.shuff(x)
        x = self.relu(x)

        x = self.conv4(x)

        return x







