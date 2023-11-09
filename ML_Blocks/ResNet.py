import torch
import torch.nn as nn

# **Heavily** Based On: https://www.youtube.com/watch?v=DkNIBBBvcPs&t=585s
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4  # number of channels after a block is 4 times that of when it entered
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample # Conv layer we do to the identity mapping so its the same shape

    def forward(self, x):
        identiy = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:    # IF we need to change the shape
            identiy = self.identity_downsample(identiy)

        x += identiy
        x = self.relu(x)
        return x
    

# block - the block described above
# layers - how many times we want to use this block; for resnet 50: [3, 4, 6, 3]
#       EX. for resnet50: [3, 4, 6, 3]
#           this means for the first resnet layer we use the block 3 times
#           then 4 times in the second layer, etc
# image_channels - number of channels of the input (RGB - 3, MNIST -1)
# num_classes - number of classes in dataset
class ResNet(nn.Module): 
    def __init__(self, block, layers, image_channels, num_classes): #may not need to send "block" in
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        #Resnet Layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # defined output size of (1,1) so AdaptiveAvGPool will pick an average pool to get that
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) #reshape so we can send it through self.fc
        x = self.fc(x)

        return x

    # This is the important shit for incorperating a custom use of the blocks (I THINK)
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # if num channels changed but identity doesnt match, we need to downsample
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*4))
        
        #this layer changes the number of channels
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks-1): # minus 1 because line 67 is the first layer
            layers.append(block(self.in_channels, out_channels))  # 256 --> _64, because output will be _64*4 which is 256 ?

        return nn.Sequential(*layers) # '*layers' unpacks the list


def ResNet50(img_channels, num_classes = 1000):
    return  ResNet(block, [3,4,6,3], img_channels, num_classes) # [3,4,6,3] what defines ResNet50

def ResNet101(img_channels, num_classes = 1000):
    return  ResNet(block, [3,4,23,3], img_channels, num_classes) # [3,4,23,3] what defines ResNet50

def ResNet152(img_channels, num_classes = 1000):
    return  ResNet(block, [3,8,36,3], img_channels, num_classes) # [3,4,6,3] what defines ResNet50


def test():
    net = ResNet152()
    x = torch.randn(2,3,224,224)
    y = net(x).to('cuda')

    print(y.shape)

test()


