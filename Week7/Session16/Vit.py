#                                                                        به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
# pip install transformers torchvision pillow datasets accelerate timm
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

# بررسی GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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

# ------------------------------------
# 2.1. بارگذاری دیتاست نمونه
# ------------------------------------
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

# ------------------------------------
# 2.2. تبدیل به Hugging Face Dataset
# ------------------------------------

def convert_to_hf_dataset(torch_dataset):
    """تبدیل torch dataset به Hugging Face dataset"""
    images = []
    labels = []
    
    for i in range(min(1000, len(torch_dataset))):  # محدود کردن برای نمونه
        image, label = torch_dataset[i]
        # تبدیل tensor به PIL Image
        image = torchvision.transforms.ToPILImage()(image)
        images.append(image)
        labels.append(label)
    
    return Dataset.from_dict({"image": images, "label": labels})

# تبدیل دیتاست‌ها
train_dataset = convert_to_hf_dataset(trainset)
test_dataset = convert_to_hf_dataset(testset)

print("Hugging Face dataset created:")
print(train_dataset)

# ------------------------------------
# 2.3. آماده‌سازی پردازنده تصویر
# ------------------------------------

# بارگذاری image processor برای ViT
processor = ViTImageProcessor.from_pretrained(VISION_MODELS['classification'])

def transform_images(examples):
    """تبدیل تصاویر برای مدل ViT"""
    images = examples['image']
    examples['pixel_values'] = processor(images, return_tensors="pt")['pixel_values']
    return examples

# اعمال تبدیل‌ها
train_dataset = train_dataset.map(transform_images, batched=True)
test_dataset = test_dataset.map(transform_images, batched=True)

print("Dataset after transformation:")
print(train_dataset[0].keys())


# --------------------------------------------------------
# 3. طبقه‌بندی تصاویر با ViT 
# --------------------------------------------------------

# ------------------------------------
# 3.1. بارگذاری مدل ViT
# ------------------------------------

# بارگذاری مدل ViT برای طبقه‌بندی
model = ViTForImageClassification.from_pretrained(
    VISION_MODELS['classification'],
    num_labels=10,  # 10 classes for CIFAR-10
    ignore_mismatched_sizes=True
)

model.to(device)
print("ViT model loaded successfully!")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------
# 3.2.  آموزش مدل ViT
# ------------------------------------

# تعریف metric برای evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir='./vit-cifar10-results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',        # تغییر این خطا
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    seed=42
)

# ایجاد Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# آموزش مدل
print("Starting training...")
train_results = trainer.train()

# ذخیره مدل
trainer.save_model()
print("Training completed!")

# ------------------------------------
# 3.3.  ارزیابی مدل
# ------------------------------------

# ارزیابی مدل
eval_results = trainer.evaluate()
print("Evaluation results:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# پیش‌بینی روی نمونه‌های تست
test_predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(test_predictions.predictions, axis=1)
true_labels = test_dataset['label']

# نمایش نمونه‌ای از پیش‌بینی‌ها
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\nSample predictions:")
for i in range(5):
    actual = class_names[true_labels[i]]
    predicted = class_names[predicted_labels[i]]
    print(f"Image {i+1}: Actual: {actual}, Predicted: {predicted}")


# --------------------------------------------------------
# 4. تشخیص اشیاء با DETR
# --------------------------------------------------------

# ------------------------------------
# 4.1 بارگذاری مدل DETR
# ------------------------------------

# بارگذاری مدل DETR برای تشخیص اشیاء
detr_processor = DetrImageProcessor.from_pretrained(VISION_MODELS['detection'])
detr_model = DetrForObjectDetection.from_pretrained(VISION_MODELS['detection'])
detr_model.to(device)

print("DETR model loaded successfully!")

# ------------------------------------
# 4.2 تشخیص اشیاء در تصاویر
# ------------------------------------
def download_image(url):
    """دانلود تصویر از URL"""
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert('RGB')

# تصویر نمونه برای تست
image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
]

def detect_objects(image):
    """تشخیص اشیاء در تصویر"""
    # پردازش تصویر
    inputs = detr_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # پیش‌بینی
    with torch.no_grad():
        outputs = detr_model(**inputs)
    
    # post-processing
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    
    return results

def draw_detections(image, detections):
    """رسم bounding boxها روی تصویر"""
    draw = ImageDraw.Draw(image)
    
    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        if score > 0.5:
            box = box.cpu().numpy()
            label_name = detr_model.config.id2label[label.item()]
            
            # رسم rectangle
            draw.rectangle(box, outline="green", width=3)
            
            # افزودن label
            draw.text((box[0], box[1]), f"{label_name}: {score:.2f}", fill="red")
    
    return image

# تست روی تصاویر نمونه
for i, url in enumerate(image_urls):
    try:
        print(f"\nProcessing image {i+1}...")
        image = download_image(url)
        
        # تشخیص اشیاء
        detections = detect_objects(image)
        
        # رسم نتایج
        result_image = draw_detections(image.copy(), detections)
        
        # نمایش نتایج
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_image)
        plt.title("Object Detection Results")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # نمایش اطلاعات تشخیص
        print("Detected objects:")
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            if score > 0.5:
                label_name = detr_model.config.id2label[label.item()]
                print(f"  {label_name}: {score:.3f}")
                
    except Exception as e:
        print(f"Error processing image {i+1}: {e}")