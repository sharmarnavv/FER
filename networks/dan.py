import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torch.nn.init as init

class DAN(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(DAN, self).__init__()
        
        # Load ConvNeXt Tiny
        # weights='DEFAULT' loads the best available weights (ImageNet)
        weights = 'DEFAULT' if pretrained else None
        try:
            self.backbone = models.convnext_tiny(weights=weights)
        except:
            print("Warning: Could not load convnext_tiny from torchvision.models. Using resnet18 as fallback to prevent crash, but this is NOT the requested architecture.")
            self.backbone = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(self.backbone.children())[:-2])
            self.in_channels = 512
        else:
            # ConvNeXt Tiny structure:
            # features: Sequential(...)
            # classifier: Sequential(LayerNorm2d, Flatten, Linear)
            # We only want the features. 
            # Output of features is (B, 768, 7, 7) for 224x224 input.
            self.features = self.backbone.features
            self.in_channels = 768 
        
        self.num_head = num_head
        
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead(self.in_channels))
        
        self.fc = nn.Linear(512, num_class) # Heads project to 512
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.features(x) # (B, 768, 7, 7)
        
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))
        
        # heads list of (B, 512) -> (B, num_head, 512)
        heads = torch.stack(heads).permute([1, 0, 2])
        
        if heads.size(1) > 1:
            # Log softmax over heads to normalize attention contributions
            heads = F.log_softmax(heads, dim=1)
            
        # Sum over heads to get aggregated feature vector: (B, 512)
        out_feat = heads.sum(dim=1)
        
        out = self.fc(out_feat)
        out = self.bn(out)
   
        return out, out_feat, heads

class CrossAttentionHead(nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()
        self.sa = SpatialAttention(in_channels)
        self.ca = ChannelAttention(in_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 768, 7, 7)
        sa = self.sa(x) # (B, 768, 7, 7) - spatially weighted
        ca = self.ca(sa) # (B, 512) - channel attended and projected
        return ca

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Reduce channels for spatial attention computation
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 768, 7, 7)
        y = self.conv1x1(x) # (B, 256, 7, 7)
        # Mix spatial features
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y)) # (B, 512, 7, 7)
        
        # Generate spatial map: sum over channels
        attn_map = y.sum(dim=1, keepdim=True) 
        
        # Apply attention to input features
        out = x * attn_map
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )
        # Projection to 512 dimension for the final feature vector
        self.projection = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, sa):
        # sa: (B, 768, 7, 7) - spatially attended features
        sa_pooled = self.gap(sa) # (B, 768, 1, 1)
        sa_vec = sa_pooled.view(sa_pooled.size(0), -1) # (B, 768)
        
        # Channel attention weights
        w = self.attention(sa_vec) # (B, 512)
        
        # Project input feature vector to 512
        x_proj = self.projection(sa_vec) # (B, 512)
        
        out = x_proj * w # (B, 512)
        
        return out

class CenterLoss(nn.Module):
    def __init__(self, num_classes=7, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / (var + 1e-6))
        else:
            loss = 0
            
        return loss
