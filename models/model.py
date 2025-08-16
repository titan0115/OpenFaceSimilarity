import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv(nn.Module):
    """Депthwise + Pointwise конволюция для MobileFaceNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                 stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu6(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        return x


class BottleneckResidual(nn.Module):
    """Остаточный блок с bottleneck для MobileFaceNet"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=2):
        super(BottleneckResidual, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                 stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.depthwise(out)))
        out = self.bn3(self.conv2(out))
        
        if self.use_residual:
            out += residual
            
        return out


class MobileFaceNet(nn.Module):
    """Легковесная модель распознавания лиц на основе MobileNet архитектуры"""
    def __init__(self, embedding_size=128, input_size=112):
        super(MobileFaceNet, self).__init__()
        self.input_size = input_size
        
        # Первый конв слой
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Depthwise конволюция
        self.dw_conv1 = DepthwiseConv(64, 64)
        
        # Последовательность остаточных блоков
        self.bottleneck1 = BottleneckResidual(64, 64, stride=2)
        self.bottleneck2 = BottleneckResidual(64, 64)
        self.bottleneck3 = BottleneckResidual(64, 64)
        self.bottleneck4 = BottleneckResidual(64, 64)
        self.bottleneck5 = BottleneckResidual(64, 64)
        
        self.bottleneck6 = BottleneckResidual(64, 128, stride=2)
        self.bottleneck7 = BottleneckResidual(128, 128)
        self.bottleneck8 = BottleneckResidual(128, 128)
        self.bottleneck9 = BottleneckResidual(128, 128)
        self.bottleneck10 = BottleneckResidual(128, 128)
        self.bottleneck11 = BottleneckResidual(128, 128)
        
        self.bottleneck12 = BottleneckResidual(128, 128, stride=2)
        self.bottleneck13 = BottleneckResidual(128, 128)
        
        # Финальные слои
        self.conv2 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        
        # Глобальный пулинг и эмбеддинг
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(512, embedding_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        # Первый блок
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.dw_conv1(x)
        
        # Остаточные блоки
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        
        # Финальные слои
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.bn3(x)
        
        # L2 нормализация для получения нормализованного эмбеддинга
        x = F.normalize(x, p=2, dim=1)
        
        return x


def count_parameters(model):
    """Подсчёт количества параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Тестирование модели
    model = MobileFaceNet(embedding_size=128)
    print(f"Количество параметров: {count_parameters(model):,}")
    
    # Тест прохода
    x = torch.randn(2, 3, 112, 112)
    embeddings = model(x)
    print(f"Размер выходного тензора: {embeddings.shape}")
    print(f"Норма эмбеддингов: {torch.norm(embeddings, p=2, dim=1)}")
