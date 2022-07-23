import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time

class ISP(nn.Module):

    def __init__(self):
        super(ISP, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
                  
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
   
        self.upv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv6_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))   
        
        up4 = self.upv4(conv3)
        up4 = torch.cat([up4, conv2], 1)
        conv4 = self.lrelu(self.conv4_1(up4))
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv1], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        conv6 = self.conv6_1(conv5)
        out = conv6
        out = nn.functional.pixel_shuffle(conv6, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

#Demoire model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.skff1 = SKFF(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.skff2 = SKFF(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
             
    def forward(self, x, y, z):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        conv_list1 = []
        conv_list1.append(conv1)
        conv_list1.append(y)
        conv_out1 = self.skff1(conv_list1)
        pool1 = self.pool1(conv_out1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        conv_list2 = []
        conv_list2.append(conv2)
        conv_list2.append(z)
        conv_out2 = self.skff2(conv_list2)
        pool2 = self.pool2(conv_out2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))   
        return conv3, conv_out2, conv_out1
        
    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

class Encoder3(nn.Module):
    def __init__(self):
        super(Encoder3, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
             
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))   
        return conv3, conv2, conv1
        
    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.ca1 = CA(128)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.ca2 = CA(64)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)   		
       
        self.conv6_1 = nn.Conv2d(64, 4, kernel_size=1, stride=1) 

    def forward(self, x, encoder_out2, encoder_out1):     
        up4 = self.upv4(x)
        up4 = torch.cat([up4, encoder_out2], 1)
        conv4 = self.lrelu(self.conv4_1(up4))
        conv4 = self.ca1(conv4)
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, encoder_out1], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.ca2(conv5)
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        conv6 = self.conv6_1(conv5)
        return conv6    

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt      

class Demoire(nn.Module):
    def __init__(self):
        super(Demoire, self).__init__()

        self.encoder1 = Encoder()    
        self.encoder2 = Encoder() 
        self.encoder3 = Encoder3()  
        self.decoder_c = Decoder() 
        self.decoder_b = Decoder()
        self.downsp1 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.downsp2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)

        self.upsp2_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsp2_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsp2_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsp3_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsp3_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsp3_3 = nn.Upsample(scale_factor=4, mode='nearest')

        # self.upsp2_1 = upsample_block2(64)
        # self.upsp2_2 = upsample_block2(128)
        # self.upsp2_3 = upsample_block2(256)
        # self.upsp3_1 = upsample_block2(64)
        # self.upsp3_2 = upsample_block2(128)
        # self.upsp3_3 = upsample_block4(256)

    def forward(self, x, flag):
        #encoder
        x1 = x
        x2 = self.downsp1(x1)
        x3 = self.downsp2(x2)
        #branch3
        encoder_out3_3, encoder_out3_2, encoder_out3_1 = self.encoder3(x3) 
        up3_1 = self.lrelu(self.upsp3_1(encoder_out3_1))
        up3_2 = self.lrelu(self.upsp3_2(encoder_out3_2)) 
        up3_3 = self.lrelu(self.upsp3_3(encoder_out3_3))
        #branch2
        encoder_out2_3, encoder_out2_2, encoder_out2_1 = self.encoder2(x2, up3_1, up3_2)
        up2_1 = self.lrelu(self.upsp2_1(encoder_out2_1)) 
        up2_2 = self.lrelu(self.upsp2_2(encoder_out2_2)) 
        up2_3 = self.lrelu(self.upsp2_3(encoder_out2_3)) 
        #branch1
        encoder_out1_3, encoder_out1_2, encoder_out1_1 = self.encoder1(x1, up2_1, up2_2) 

        #encoder out
        #print(encoder_out3_1.shape)
        #print(encoder_out3_2.shape)
        #print(encoder_out3_3.shape)
        sk_out1 = encoder_out1_1
        sk_out2 = encoder_out1_2
        out = encoder_out1_3 + up2_3 + up3_3

        #decoder
        if flag == 1:
            out_c = self.decoder_c(out, sk_out2, sk_out1)
            return out_c

        if flag == 0:
            out_b = self.decoder_b(out, sk_out2, sk_out1)
            return out_b
        
    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt
        
# class upsample_block2(nn.Module):
#     def __init__(self, in_channels):
#         super(upsample_block2, self).__init__()
#         self.conv = nn.Conv2d(in_channels, 4*in_channels,
#                               3, stride=1, padding=1)
#         self.shuffler = nn.PixelShuffle(2)

#     def forward(self, x):
#         return self.shuffler(self.conv(x))

# class upsample_block4(nn.Module):
#     def __init__(self, in_channels):
#         super(upsample_block4, self).__init__()
#         self.conv = nn.Conv2d(in_channels, 16*in_channels,
#                               3, stride=1, padding=1)
#         self.shuffler = nn.PixelShuffle(4)

#     def forward(self, x):
#         return self.shuffler(self.conv(x))

# class channel_attention(nn.Module):
#     def __init__(self, in_channels):
#         super(channel_attention, self).__init__()
#         self.ave_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(in_channels, in_channels//16, 1, stride=1)
#         self.conv2 = nn.Conv2d(in_channels//16, in_channels, 1, stride=1)
        
#     def forward(self, x):
#         return self.shuffler(self.conv(x))

class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V  