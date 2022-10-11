# Define network components here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import models.losses as losses
import models.pytorch_colors as colors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)#.cuda()


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        

class H_Attention(nn.Module):
    def __init__(self, in_dim):
        super(H_Attention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)
        #self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)

        energy_H = self.softmax(torch.bmm(proj_query_H, proj_key_H))# + self.INF(m_batchsize, height, width))
        out_H = torch.bmm(proj_value_H, energy_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        return self.gamma * out_H + x

class W_Attention(nn.Module):
    def __init__(self, in_dim):
        super(W_Attention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)
        #self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,1)

        proj_key = self.key_conv(x)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_W = self.softmax(torch.bmm(proj_query_W, proj_key_W))
        out_W = torch.bmm(proj_value_W, energy_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * out_W+ x

class R3Attention(nn.Module):
    def __init__(self, in_dim,residual = False):
        super(R3Attention,self).__init__()
        self.H_Attention1 = H_Attention(in_dim)
        self.W_Attention = W_Attention(in_dim)
        self.H_Attention2 = H_Attention(in_dim)
        self.residual = residual
    def forward(self, x):
        H_x = self.H_Attention1(x)
        WH_x = self.W_Attention(H_x)
        HWH_x = self.H_Attention2(WH_x)
        if self.residual == True:
            HWH_x = x + HWH_x
        return HWH_x

class Dalations_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, mode=None):
        super(Dalations_ResBlock, self).__init__()
        feature = in_channels
        self.conv1 = nn.Conv2d(in_channels, feature, kernel_size=3, padding=1, bias=bias)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2_1 = nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=bias)
        self.conv2_2 = nn.Conv2d(feature, feature, kernel_size=3, padding=3, bias=bias,dilation=3)
        self.conv2_3 = nn.Conv2d(feature, feature, kernel_size=3, padding=5, bias=bias,dilation=5)
        self.conv3 = nn.Conv2d((feature*3), out_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual1 = self.relu1(self.conv2_1(residual))
        residual2 = self.relu1(self.conv2_2(residual))
        residual3 = self.relu1(self.conv2_3(residual))
        residual = torch.cat((residual1, residual2, residual3), dim=1)
        out = self.conv3(residual)
        return x +out

class GL_context_encoding(nn.Module):
    def __init__(self, channels, bias=True, mode=None):
        super(GL_context_encoding, self).__init__()
        feature = channels
        self.Dalations_ResBlock = Dalations_ResBlock(in_channels=channels, out_channels=channels)
        self.R3Attention = R3Attention(in_dim=channels,residual = False)
        self.conv_tail = nn.Conv2d((feature*2), feature, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        fea1 = self.Dalations_ResBlock(x)
        fea2 = self.R3Attention(x)
        fea = torch.cat([fea1,fea2],dim=1)
        out = self.conv_tail(fea)
        return out + x

class GL_context_encoding1(nn.Module):
    def __init__(self, channels,dilation=1, norm=None, act=nn.ReLU(True), se_reduction=None, res_scale=1,bias=True, mode=None):
        super(GL_context_encoding1, self).__init__()
        self.R3Attention = R3Attention(in_dim=channels,residual = False)
        self.conv_tail = ResidualBlock(
            channels, dilation=dilation, norm=norm, act=act,
            se_reduction=se_reduction, res_scale=res_scale)

    def forward(self, x):
        fea = self.R3Attention(x)
        out = self.conv_tail(fea)
        return out + x

class DRNet1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False,hyper=True):
        super(DRNet1, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.head_H = nn.Sequential(*[
            ConvLayer(conv, in_channels + 1, n_feats // 2, kernel_size=bottom_kernel_size, stride=1, norm=None,
                      act=act),
            ConvLayer(conv, n_feats // 2, n_feats // 2, kernel_size=3, stride=1, norm=norm, act=act),
            ConvLayer(conv, n_feats // 2, n_feats // 2, kernel_size=3, stride=2, norm=norm, act=act)
        ])

        self.hyper = hyper
        if self.hyper:
            self.vgg = losses.Vgg19(requires_grad=False)
            in_channels += 1472
        
        self.pyramid_module = None
        self.head = nn.Sequential(*[
            ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act),
            ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act),
            ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)
        ])


        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.ModuleList([
            nn.Sequential(*[GL_context_encoding(n_feats) for i in range(n_resblocks)]),
            nn.Sequential(*[GL_context_encoding(n_feats) for i in range(n_resblocks)]),
            nn.Sequential(*[GL_context_encoding(n_feats) for i in range(n_resblocks)])
            ])

        self.res_module_H = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(
                    n_feats//2, dilation=dilation_config[i], norm=norm, act=act,
                    se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)]),
            nn.Sequential(*[ResidualBlock(
                    n_feats//2, dilation=dilation_config[i], norm=norm, act=act,
                    se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)]),
            nn.Sequential(*[ResidualBlock(
                n_feats//2, dilation=dilation_config[i], norm=norm, act=act,
                se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])
        ])

        self.convs_1x1 = nn.ModuleList([
            ConvLayer(conv, n_feats + n_feats//2, n_feats, kernel_size=1, stride=1, norm=norm, act=act),
            ConvLayer(conv, n_feats + n_feats//2, n_feats, kernel_size=1, stride=1, norm=norm, act=act),
            ConvLayer(conv, n_feats + n_feats//2, n_feats, kernel_size=1, stride=1, norm=norm, act=act)
        ])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        self.deconv1_h = ConvLayer(deconv, n_feats//2, n_feats//2, kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv2_h = ConvLayer(conv, n_feats//2, n_feats//2, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
            self.deconv3_h = ConvLayer(conv, n_feats//2, 1, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv2_h = ConvLayer(conv, n_feats // 2, n_feats // 2, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
            self.deconv3_h = ConvLayer(conv, n_feats // 2, 1, kernel_size=1, stride=1, norm=None, act=act)

    def forward(self, x):

        input_hsv = colors.rgb_to_hsv(x.detach())  #
        input_hsv_h = torch.unsqueeze(input_hsv[:, 0, :, :], 1)

        if self.hyper:
            hypercolumn = self.vgg(x)
            _, C, H, W = x.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                           feature in hypercolumn]
            input_x = [x]
            input_x.extend(hypercolumn)
            input_x = torch.cat(input_x, dim=1)
            input_h_x = torch.cat([input_hsv_h, x], dim=1)
        else:
            input_x = x
            input_h_x = torch.cat([input_hsv_h, x], dim=1)

        x_fea = self.head(input_x) #这一步downsample了
        h_fea = self.head_H(input_h_x)

        x0_h = self.res_module_H[0](h_fea)
        x1_h = self.res_module_H[1](x0_h)
        x2_h = self.res_module_H[2](x1_h)

        x0 = self.res_module[0](x_fea)
        x0 = self.convs_1x1[0]( torch.cat([x0,x0_h], dim=1) )

        x1 = self.res_module[1](x0)
        x1 = self.convs_1x1[1]( torch.cat([x1, x1_h], dim=1))

        x2 = self.res_module[2](x1)
        x2 = self.convs_1x1[2](torch.cat([x2, x2_h], dim=1))


        x2 = self.deconv1(x2)#这一步upsample了

        x2 = self.deconv2(x2)
        if self.pyramid_module is not None:
            x2 = self.pyramid_module(x2)
        x2 = self.deconv3(x2)

        x2_h = self.deconv1_h(x2_h)  # 这一步upsample了
        x2_h = self.deconv2_h(x2_h)
        x2_h = self.deconv3_h(x2_h)


        return x2,x2_h


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=None, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)


if __name__ == "__main__":
    x = torch.randn(2, 128, 128, 128)
    model = PyramidPooling(128,128)
    out = model(x)
    print(out.shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))

    x = torch.randn(2, 3, 128, 128)        #.cuda()#.to(torch.device('cuda'))
    fbar=DRNet1(in_channels=3, out_channels=3, n_feats= 144, n_resblocks=4,
               norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,hyper=True)
    # print(fbar)  之前13 resblocks-256 18576547  现在 6个
    y,y_h = fbar(x)
    print(y.shape,y_h.shape)
    print('-' * 50)
    print('#generator parameters:', sum(param.numel() for param in fbar.parameters()))
