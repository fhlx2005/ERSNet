import torch.nn
from fightingcv_attention.attention.MobileViTAttention import MobileViTAttention, Transformer
from fightingcv_attention.attention.ShuffleAttention import ShuffleAttention
from torch import nn
from thop import profile
from torchsummary import summary
from einops import rearrange


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CBAMChannel(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMChannel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class CBAMSpatial(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(CBAMSpatial, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = CBAMChannel(in_channels, reduction)
        self.SpatialGate = CBAMSpatial(kernel_size)

    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        return x


def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
        return nn.Sequential(
           
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

  
def Conv1x1BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

   
def Conv1x1BN(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

class InvertedResidual(nn.Module):
    
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)
        self.expansion_factor = expansion_factor
        
        self.bottleneck = nn.Sequential(
            Conv3x3BNReLU(in_channels, mid_channels, stride, groups=in_channels),
            Conv1x1BN(mid_channels, out_channels),
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)


    def forward(self, x):
        out = self.bottleneck(x)
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out

class ERSNet(nn.Module):
    def __init__(self, in_channel, n_class, t=6):
        super(ERSNet, self).__init__()
        self.in_chanel = in_channel
        self.n_class = n_class

        self.layer1_3_1 = torch.nn.Conv2d(in_channels=self.in_chanel, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.layer1_3_d2 = torch.nn.Conv2d(in_channels=self.in_chanel, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.layer1_5_d5 = torch.nn.Conv2d(in_channels=self.in_chanel, out_channels=32, kernel_size=7, stride=2, padding=3)


        self.cbam1 = CBAM(in_channels=32)
        
        self.layer2 = self.make_layer(in_channels=32, out_channels=64, stride=1, factor=4, block_num=4) #32*32*64 （4，6，4）
        self.cbam2 = CBAM(in_channels=64)
        
        self.layer3 = self.make_layer(in_channels=64, out_channels=96, stride=2, factor=6, block_num=4) # 16*16* 96
        self.cbam3 = CBAM(in_channels=96)
       

        self.layer5 = self.make_layer(in_channels=96, out_channels=192, stride=2, factor=4, block_num=3)
        self.cbam5 = CBAM(in_channels=192)

        self.layer6 = self.make_layer(in_channels=192, out_channels=384, stride=2, factor=4, block_num=2)
        self.cbam6 = CBAM(in_channels=384)

        self.se = ShuffleAttention(channel=384, G=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(  
            nn.Dropout(0.2),
            nn.Linear(384, self.n_class)
        )

    def make_layer(self, in_channels, out_channels, stride, factor, block_num):
        layers = []
      
        layers.append(InvertedResidual(in_channels, out_channels, factor, stride))
       
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, factor, 1))
        return nn.Sequential(*layers)



    def forward(self, x):
       x_1 = self.layer1_3_1(x)
       x_2 = self.layer1_3_d2(x)
       x_3 = self.layer1_5_d5(x)
       x = x_1 + x_2 + x_3   #/ (1/2)H (1/2)W,
       x = self.cbam1(x)
       x = self.layer2(x)
       x = self.cbam2(x)
       x = self.layer3(x)
       x = self.cbam3(x)
       x = self.layer5(x)
       x = self.cbam5(x)
       x = self.layer6(x)
       x = self.cbam6(x)
       x = self.se(x)
       x = self.avgpool(x)
       x = torch.flatten(x,1)
       x = self.classifier(x)
       return x


if __name__ == '__main__':
     
    data = torch.randn(1, 3, 64, 64).to('cuda')
    module = ERSNet(in_channel=3, n_class=10)
    # data.to(DEVICE)
    module.to(DEVICE).eval()
    print(summary(module, (3, 64, 64), device='cuda'))
    flops, params = profile(module, (data,))
    print('FLOPs(G)', flops/(1000**3), 'params', params/1000000.0)












