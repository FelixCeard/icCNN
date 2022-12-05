import torch
import torch.nn as nn


class EMA_FM():
    def __init__(self, decay=0.9, first_decay=0.0, channel_size=512, f_map_size=196, is_use=False):
        self.decay = decay
        self.first_decay = first_decay
        self.is_use = is_use
        self.shadow = {}
        self.epsional = 1e-5
        if is_use:
            self._register(channel_size=channel_size, f_map_size=f_map_size)

    def _register(self, channel_size=512, f_map_size=196):
        Init_FM = torch.zeros((f_map_size, channel_size), dtype=torch.float)
        self.shadow['FM'] = Init_FM.cuda().clone()
        self.is_first = True

    def update(self, input):
        B, C, _ = input.size()
        if not (self.is_use):
            return torch.ones((C, C), dtype=torch.float)
        decay = self.first_decay if self.is_first else self.decay
        ####### FEATURE SIMILARITY MATRIX EMA ########
        # Mu = torch.mean(input,dim=0)
        self.shadow['FM'] = (1.0 - decay) * torch.mean(input, dim=0) + decay * self.shadow['FM']
        self.is_first = False
        return self.shadow['FM']


class SMGBlock(nn.Module):
    def __init__(self, channel_size=2048, f_map_size=196):
        super(SMGBlock, self).__init__()

        self.EMA_FM = EMA_FM(decay=0.95, first_decay=0.0, channel_size=channel_size, f_map_size=f_map_size, is_use=True)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        batch_size, channel, _, _ = x.size()
        theta_x = x.view(batch_size, channel, -1).permute(0, 2, 1).contiguous()
        transpose_x = x.view(batch_size, channel, -1).permute(0, 2, 1).contiguous()  # [b,h×w,c]
        with torch.no_grad():
            f_mean = self.EMA_FM.update(theta_x)
        sz = f_mean.size()[0]
        f_mean = f_mean.view(1, channel, sz)
        f_mean_transposed = f_mean.permute(0, 2, 1)
        Local = torch.matmul(theta_x.permute(0, 2, 1) - f_mean, theta_x - f_mean_transposed)
        diag = torch.eye(channel).view(-1, channel, channel).cuda()
        cov = torch.sum(Local * diag, dim=2).view(batch_size, channel, 1)
        cov_transpose = cov.permute(0, 2, 1)
        norm = torch.sqrt(torch.matmul(cov, cov_transpose))
        correlation = torch.div(Local, norm) + 1  ## normlize to [0,2]

        return correlation


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1)  # new paddig

    def forward(self, x):
        identity = x
        out = self.pad2d(x)  # new padding
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out)  # new padding
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1)  # new paddig

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out)  # new padding
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class newPad2d(nn.Module):
    def __init__(self, length):
        super(newPad2d, self).__init__()
        self.length = length
        self.zeroPad = nn.ZeroPad2d(self.length)

    def forward(self, input):
        b, c, h, w = input.shape
        output = self.zeroPad(input)

        # output = torch.FloatTensor(b,c,h+self.length*2,w+self.length*2)
        # output[:,:,self.length:self.length+h,self.length:self.length+w] = input

        for i in range(self.length):
            # 一层的四个切片
            output[:, :, self.length:self.length + h, i] = output[:, :, self.length:self.length + h, self.length]
            output[:, :, self.length:self.length + h, w + self.length + i] = output[:, :, self.length:self.length + h,
                                                                             self.length - 1 + w]
            output[:, :, i, self.length:self.length + w] = output[:, :, self.length, self.length:self.length + w]
            output[:, :, h + self.length + i, self.length:self.length + w] = output[:, :, h + self.length - 1,
                                                                             self.length:self.length + w]
        # 对角进行特别处理
        for j in range(self.length):
            for k in range(self.length):
                output[:, :, j, k] = output[:, :, self.length, self.length]
                output[:, :, j, w + self.length + k] = output[:, :, self.length, self.length - 1 + w]
                output[:, :, h + self.length + j, k] = output[:, :, h + self.length - 1, self.length]
                output[:, :, h + self.length + j, w + self.length + k] = output[:, :, h + self.length - 1,
                                                                         self.length - 1 + w]
        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)  # new padding


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
