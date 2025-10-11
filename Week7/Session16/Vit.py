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
            draw.rectangle(box, outline="Chartreuse", width=3)
            
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

# --------------------------------------------------------
# 5. سیستم کامل بینایی کامپیوتر 
# --------------------------------------------------------

# ------------------------------------
# 5.1 ایجاد سیستم بینایی کامپیوتر چندمنظوره
# ------------------------------------

class ComputerVisionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
    
    def _load_models(self):
        """بارگذاری همه مدل‌های بینایی کامپیوتر"""
        print("Loading computer vision models...")
        
        # مدل طبقه‌بندی تصاویر
        self.classification_processor = ViTImageProcessor.from_pretrained(VISION_MODELS['classification'])
        self.classification_model = ViTForImageClassification.from_pretrained(
            VISION_MODELS['classification']
        )
        self.classification_model.to(self.device)
        
        # مدل تشخیص اشیاء
        self.detection_processor = DetrImageProcessor.from_pretrained(VISION_MODELS['detection'])
        self.detection_model = DetrForObjectDetection.from_pretrained(VISION_MODELS['detection'])
        self.detection_model.to(self.device)
        
        print("All models loaded successfully!")
    
    def analyze_image(self, image):
        """آنالیز کامل تصویر"""
        results = {}
        
        # طبقه‌بندی تصویر
        results['classification'] = self._classify_image(image)
        
        # تشخیص اشیاء
        results['detection'] = self._detect_objects(image)
        
        # استخراج features
        results['features'] = self._extract_features(image)
        
        return results
    
    def _classify_image(self, image):
        """طبقه‌بندی تصویر"""
        inputs = self.classification_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.classification_model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)
        
        # گرفتن top-3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            label = self.classification_model.config.id2label[top3_indices[0][i].item()]
            prob = top3_probs[0][i].item()
            predictions.append({'label': label, 'confidence': prob})
        
        return {
            'top_prediction': predictions[0],
            'all_predictions': predictions
        }
    
    def _detect_objects(self, image):
        """تشخیص اشیاء در تصویر"""
        inputs = self.detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.3:
                label_name = self.detection_model.config.id2label[label.item()]
                detections.append({
                    'label': label_name,
                    'confidence': score.item(),
                    'bbox': box.cpu().numpy().tolist()
                })
        
        return detections
    
    def _extract_features(self, image):
        """استخراج features از تصویر"""
        inputs = self.classification_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.classification_model(**inputs, output_hidden_states=True)
        
        # گرفتن features از آخرین لایه
        features = outputs.hidden_states[-1][0, 0, :].cpu().numpy()
        
        return {
            'feature_vector': features.tolist(),
            'feature_dim': len(features)
        }
    
    def visualize_analysis(self, image, analysis):
        """تجسم نتایج آنالیز"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # تصویر اصلی
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # تصویر با bounding boxها
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        for detection in analysis['detection']:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            # رسم bounding box
            draw.rectangle(bbox, outline='Chartreuse', width=3)
            draw.text((bbox[0], bbox[1]), f"{label}: {confidence:.2f}", fill='red')
        
        axes[1].imshow(result_image)
        axes[1].set_title('Object Detection Results')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # نمایش نتایج متنی
        print("\n=== IMAGE ANALYSIS RESULTS ===")
        print(f"Top classification: {analysis['classification']['top_prediction']['label']} "
              f"(confidence: {analysis['classification']['top_prediction']['confidence']:.3f})")
        
        print("\nAll classifications:")
        for pred in analysis['classification']['all_predictions']:
            print(f"  {pred['label']}: {pred['confidence']:.3f}")
        
        print(f"\nDetected objects ({len(analysis['detection'])}):")
        for obj in analysis['detection']:
            print(f"  {obj['label']}: {obj['confidence']:.3f}")
        
        print(f"\nFeature vector dimension: {analysis['features']['feature_dim']}")

# ایجاد و تست سیستم
cv_system = ComputerVisionSystem()

# تست با تصویر نمونه
try:
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    test_image = download_image(test_image_url)
    
    print("Analyzing image...")
    analysis = cv_system.analyze_image(test_image)
    cv_system.visualize_analysis(test_image, analysis)
    
except Exception as e:
    print(f"Error in analysis: {e}")
    
    # تست با تصویر محلی (fallback)
    print("Trying with a simple created image...")
    # ایجاد یک تصویر ساده برای تست
    test_image = Image.new('RGB', (224, 224), color='lightblue')
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
    
    analysis = cv_system.analyze_image(test_image)
    cv_system.visualize_analysis(test_image, analysis)