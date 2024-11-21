import torch
import torch.nn as nn
from torchvision import models

class TimeEmbedding_2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(1, dim, dtype=torch.float16)
        self.linear2 = nn.Linear(dim, dim, dtype=torch.float16)
        self.activation = nn.GELU()

    def forward(self, t):
        t = t.half().view(-1, 1)
        t = self.activation(self.linear1(t))
        t = self.linear2(t)
        return t

class EnhancedResNetLatentClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, time_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding_2(time_dim)
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first conv layer to accept input_channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the last FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Convert ResNet to half precision
        self.resnet = self.resnet.half()
        
        # Add new layers
        self.fc1 = nn.Linear(2048 + time_dim, 1024, dtype=torch.float16)
        self.fc2 = nn.Linear(1024, 512, dtype=torch.float16)
        self.fc3 = nn.Linear(512, num_classes, dtype=torch.float16)
        
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.activation = nn.GELU()

    def forward(self, x, t):
        x = x.half()
        t = t.half()
        
        # ResNet feature extraction
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Time embedding
        t = self.time_embed(t)
        
        # Ensure t has the same batch size as x
        t = t.expand(x.size(0), -1)
        
        # Concatenate features and time embedding
        x = torch.cat([x, t], dim=1)
        
        # Final classification layers with residual connections
        residual = x
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = x + residual[:, :1024]  # Residual connection
        
        residual = x
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual[:, :512]  # Residual connection
        
        x = self.fc3(x)
        
        return x

def get_classifier(dataset_name, input_channels, num_classes):
    if dataset_name.lower() == "cifar10":
        return EnhancedResNetLatentClassifier(input_channels, num_classes)
    elif dataset_name.lower() == "cifar100":
        return EnhancedResNetLatentClassifier(input_channels, num_classes)
    elif dataset_name.lower() == "tinyimagenet":
        return EnhancedResNetLatentClassifier(input_channels, num_classes)
    elif dataset_name.lower() == "imagenet100":
        return EnhancedResNetLatentClassifier(input_channels, num_classes)
    elif dataset_name.lower() == "imagenet1k":
        return EnhancedResNetLatentClassifier(input_channels, num_classes)
    elif dataset_name.lower() == "imagewoof":
        return EnhancedResNetLatentClassifier(input_channels, num_classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
