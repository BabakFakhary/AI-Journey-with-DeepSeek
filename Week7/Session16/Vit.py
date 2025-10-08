#                                                                        به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install transformers torchvision pillow datasets accelerate
import torch
import torchvision
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    DetrImageProcessor, 
    DetrForObjectDetection,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, load_dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# ==================================================================================================
#  یک مدل یادگیری عمیق است که از معماری **ترانسفورمر**  برای وظایف پردازش تصویر استفاده می‌کند.   
# ==================================================================================================


# --------------------------------------------------------
# 1. معرفی مدل‌های بینایی کامپیوتر
# --------------------------------------------------------
VISION_MODELS = {
    'classification': 'google/vit-base-patch16-224',
    'detection': 'facebook/detr-resnet-50'
}

print("Available Vision Transformer models:")
for task, model_name in VISION_MODELS.items():
    print(f"{task.upper():15}: {model_name}")

# --------------------------------------------------------
# 2. آماده‌سازی داده تصویری
# --------------------------------------------------------
# استفاده از دیتاست CIFAR-10 از torchvision

def load_cifar10_dataset():
    """بارگذاری دیتاست CIFAR-10"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    return trainset, testset

# بارگذاری دیتاست
trainset, testset = load_cifar10_dataset()

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
print(f"Number of classes: {len(trainset.classes)}")
print(f"Classes: {trainset.classes}")