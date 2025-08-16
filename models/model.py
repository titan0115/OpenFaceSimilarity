import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-Excitation блок для механизма внимания"""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CoordinateAttention(nn.Module):
    """Coordinate Attention - учитывает как канальные, так и пространственные зависимости"""
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Пулинг по высоте и ширине
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Объединение
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Разделение и получение весов внимания
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h


class GeMPooling(nn.Module):
    """Generalized Mean Pooling - более эффективная альтернатива AdaptiveAvgPool2d"""
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1./self.p)


class DepthwiseConv(nn.Module):
    """Депthwise + Pointwise конволюция для MobileFaceNet (оригинальная версия)"""
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


class DepthwiseConvV3(nn.Module):
    """Улучшенная Depthwise + Pointwise конволюция с h-swish активацией"""
    def __init__(self, in_channels, out_channels, stride=1, use_hardswish=True):
        super(DepthwiseConvV3, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                 stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_hardswish = use_hardswish
        
    def forward(self, x):
        if self.use_hardswish:
            x = F.hardswish(self.bn1(self.depthwise(x)))
        else:
            x = F.relu6(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        return x


class BottleneckResidual(nn.Module):
    """Остаточный блок с bottleneck для MobileFaceNet (оригинальная версия)"""
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


class BottleneckResidualV3(nn.Module):
    """Улучшенный остаточный блок с SE-модулем, h-swish активацией и опциональным Coordinate Attention"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=2, 
                 use_se=True, use_ca=False, se_reduction=4):
        super(BottleneckResidualV3, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        self.use_ca = use_ca

        # Первая 1x1 свертка (расширение)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Depthwise 3x3 свертка
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                 stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # SE или Coordinate Attention блок
        if use_se and not use_ca:
            self.attention = SELayer(hidden_dim, reduction=se_reduction)
        elif use_ca:
            self.attention = CoordinateAttention(hidden_dim)
        else:
            self.attention = None

        # Финальная 1x1 свертка (сжатие)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        
        # Первая свертка с h-swish активацией
        out = F.hardswish(self.bn1(self.conv1(x)))
        
        # Depthwise свертка с h-swish активацией
        out = F.hardswish(self.bn2(self.depthwise(out)))
        
        # Применение модуля внимания
        if self.attention is not None:
            out = self.attention(out)
        
        # Финальная свертка без активации
        out = self.bn3(self.conv2(out))
        
        # Остаточное соединение
        if self.use_residual:
            out += residual
            
        return out


class FusedMBConv(nn.Module):
    """Fused MBConv блок из EfficientNetV2 - объединяет 3x3 и 1x1 свертки для повышения эффективности"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, use_se=True):
        super(FusedMBConv, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        
        # Объединенная 3x3 свертка (заменяет 1x1 + 3x3)
        self.fused_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, 
                                   stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # SE блок
        if use_se:
            self.se = SELayer(hidden_dim)
        
        # Финальная 1x1 свертка
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        
        # Объединенная свертка с h-swish
        out = F.hardswish(self.bn1(self.fused_conv(x)))
        
        # SE модуль
        if self.use_se:
            out = self.se(out)
        
        # Проекционная свертка
        out = self.bn2(self.project_conv(out))
        
        # Остаточное соединение
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


class MobileFaceNetV3(nn.Module):
    """Модернизированная модель распознавания лиц с SE/CA блоками, h-swish активацией и GeM-пулингом"""
    def __init__(self, embedding_size=128, input_size=112, use_gem_pooling=True, 
                 attention_type='se', block_type='v3'):
        super(MobileFaceNetV3, self).__init__()
        self.input_size = input_size
        self.use_gem_pooling = use_gem_pooling
        self.attention_type = attention_type  # 'se', 'ca', or None
        
        # Первый конв слой с h-swish
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Улучшенная Depthwise конволюция
        self.dw_conv1 = DepthwiseConvV3(64, 64, use_hardswish=True)
        
        # Выбор типа блока
        if block_type == 'v3':
            block_class = BottleneckResidualV3
        elif block_type == 'fused':
            block_class = FusedMBConv
        else:
            block_class = BottleneckResidual
            
        use_se = attention_type == 'se'
        use_ca = attention_type == 'ca'
        
        # Последовательность модернизированных остаточных блоков
        # Первая группа (64 каналов)
        self.bottleneck1 = block_class(64, 64, stride=2, use_se=use_se, use_ca=use_ca)
        self.bottleneck2 = block_class(64, 64, use_se=use_se, use_ca=use_ca)
        self.bottleneck3 = block_class(64, 64, use_se=use_se, use_ca=use_ca)
        self.bottleneck4 = block_class(64, 64, use_se=use_se, use_ca=use_ca)
        self.bottleneck5 = block_class(64, 64, use_se=use_se, use_ca=use_ca)
        
        # Вторая группа (128 каналов) - более важные блоки с SE/CA
        self.bottleneck6 = block_class(64, 128, stride=2, use_se=True, use_ca=use_ca)
        self.bottleneck7 = block_class(128, 128, use_se=True, use_ca=use_ca)
        self.bottleneck8 = block_class(128, 128, use_se=True, use_ca=use_ca)
        self.bottleneck9 = block_class(128, 128, use_se=True, use_ca=use_ca)
        self.bottleneck10 = block_class(128, 128, use_se=True, use_ca=use_ca)
        self.bottleneck11 = block_class(128, 128, use_se=True, use_ca=use_ca)
        
        # Третья группа (256 каналов) - обязательно с вниманием
        self.bottleneck12 = block_class(128, 256, stride=2, use_se=True, use_ca=use_ca)
        self.bottleneck13 = block_class(256, 256, use_se=True, use_ca=use_ca)
        
        # Финальные слои с h-swish
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        
        # Выбор типа пулинга
        if use_gem_pooling:
            self.global_pool = GeMPooling()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
        self.conv3 = nn.Conv2d(512, embedding_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        # Первый блок с h-swish
        x = F.hardswish(self.bn1(self.conv1(x)))
        x = self.dw_conv1(x)
        
        # Первая группа остаточных блоков
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        
        # Вторая группа остаточных блоков  
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        
        # Третья группа остаточных блоков
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        
        # Финальные слои с h-swish
        x = F.hardswish(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.bn3(x)
        
        # L2 нормализация для получения нормализованного эмбеддинга
        x = F.normalize(x, p=2, dim=1)
        
        return x


class MobileFaceNetEfficient(nn.Module):
    """Версия с EfficientNetV2 блоками для максимальной производительности"""
    def __init__(self, embedding_size=128, input_size=112):
        super(MobileFaceNetEfficient, self).__init__()
        self.input_size = input_size
        
        # Первый конв слой
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Комбинация Fused и обычных MBConv блоков
        self.fused_block1 = FusedMBConv(32, 64, stride=1, expansion=1, use_se=False)
        self.fused_block2 = FusedMBConv(64, 64, stride=2, expansion=4, use_se=False)
        
        # Переход к обычным MBConv с SE
        self.mb_block1 = BottleneckResidualV3(64, 128, stride=2, use_se=True)
        self.mb_block2 = BottleneckResidualV3(128, 128, use_se=True)
        self.mb_block3 = BottleneckResidualV3(128, 128, use_se=True)
        self.mb_block4 = BottleneckResidualV3(128, 128, use_se=True)
        
        self.mb_block5 = BottleneckResidualV3(128, 256, stride=2, use_se=True)
        self.mb_block6 = BottleneckResidualV3(256, 256, use_se=True)
        
        # Финальные слои
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.global_pool = GeMPooling()
        self.conv3 = nn.Conv2d(512, embedding_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = F.hardswish(self.bn1(self.conv1(x)))
        
        # Fused блоки для начальных слоев
        x = self.fused_block1(x)
        x = self.fused_block2(x)
        
        # Обычные MBConv блоки с SE
        x = self.mb_block1(x)
        x = self.mb_block2(x)
        x = self.mb_block3(x)
        x = self.mb_block4(x)
        x = self.mb_block5(x)
        x = self.mb_block6(x)
        
        # Финальные слои
        x = F.hardswish(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.bn3(x)
        
        x = F.normalize(x, p=2, dim=1)
        return x


def count_parameters(model):
    """Подсчёт количества параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 80)
    print("СРАВНЕНИЕ АРХИТЕКТУР MOBILEFACENET")
    print("=" * 80)
    
    # Тестирование оригинальной модели
    print("\n1. Оригинальная MobileFaceNet:")
    model_orig = MobileFaceNet(embedding_size=128)
    print(f"   Параметры: {count_parameters(model_orig):,}")
    
    # Тестирование модернизированной модели с SE
    print("\n2. MobileFaceNetV3 с SE блоками:")
    model_v3_se = MobileFaceNetV3(embedding_size=128, attention_type='se', use_gem_pooling=True)
    print(f"   Параметры: {count_parameters(model_v3_se):,}")
    
    # Тестирование модернизированной модели с Coordinate Attention
    print("\n3. MobileFaceNetV3 с Coordinate Attention:")
    model_v3_ca = MobileFaceNetV3(embedding_size=128, attention_type='ca', use_gem_pooling=True)
    print(f"   Параметры: {count_parameters(model_v3_ca):,}")
    
    # Тестирование EfficientNet версии
    print("\n4. MobileFaceNetEfficient:")
    model_efficient = MobileFaceNetEfficient(embedding_size=128)
    print(f"   Параметры: {count_parameters(model_efficient):,}")
    
    # Тест прохода всех моделей
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ ПРОХОДА ДАННЫХ")
    print("=" * 50)
    
    x = torch.randn(2, 3, 112, 112)
    print(f"Входной тензор: {x.shape}")
    
    models = {
        "Оригинальная": model_orig,
        "V3 + SE": model_v3_se, 
        "V3 + CA": model_v3_ca,
        "Efficient": model_efficient
    }
    
    for name, model in models.items():
        try:
            with torch.no_grad():
                embeddings = model(x)
                norm = torch.norm(embeddings, p=2, dim=1)
                print(f"{name:15}: {embeddings.shape} | Норма: {norm.mean():.6f}")
        except Exception as e:
            print(f"{name:15}: ОШИБКА - {e}")
    
    print("\n" + "=" * 80)
    print("ПРЕИМУЩЕСТВА НОВЫХ АРХИТЕКТУР:")
    print("- SE блоки: улучшенное внимание к важным каналам")
    print("- Coordinate Attention: учет пространственной информации")  
    print("- h-swish активация: более эффективная чем ReLU6")
    print("- GeM пулинг: адаптивное усреднение/максимизация")
    print("- Fused блоки: повышенная эффективность на современных ускорителях")
    print("=" * 80)
