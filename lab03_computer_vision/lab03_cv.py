import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import cv2
import numpy as np
from pathlib import Path

# Пристрій для обчислень
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Трансформації для pretrained-моделей (ImageNet стандарти)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Завантаження попередньо навченої моделі (Transfer Learning)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Feature Extraction: заморожуємо базові шари
for param in model.parameters():
    param.requires_grad = False

# Замінюємо фінальний шар під 3 класи (можеш змінити під свій датасет)
num_classes = 3
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)
model.eval()

# 3. Бенчмарк Inference Latency та FPS (після warm-up) на 100 ітераціях
print("Вимірювання продуктивності (Benchmark)...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Warm-up (прогрів)
for _ in range(10):
    _ = model(dummy_input)

num_iterations = 100
start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(num_iterations):
        _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

end_time = time.perf_counter()
total_time = end_time - start_time
avg_latency_ms = (total_time / num_iterations) * 1000
fps = num_iterations / total_time

print(f"Середній час inference (Latency): {avg_latency_ms:.2f} мс")
print(f"Пропускна здатність (Throughput / FPS): {fps:.2f} кадров/сек")

# 4. OpenCV-конвеєр для обробки зображення/кадру
# Шлях до зображення відносно розташування цього скрипту
image_path = Path(__file__).resolve().parent / "ziba.jpg"
if not image_path.is_file():
    print(f"Помилка: не вдалося завантажити зображення за шляхом: {image_path}")
    raise FileNotFoundError(image_path)

# Читання байтів через NumPy обходить проблеми cv2.imread з Unicode-шляхами у Windows.
image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
if frame is None:
    print(f"Помилка: не вдалося завантажити зображення за шляхом: {image_path}")
    raise ValueError(f"Не вдалося декодувати зображення: {image_path}")

t_start = time.perf_counter()

# Preprocessing
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
img_tensor = transform(img_resized).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)

t_end = time.perf_counter()
full_pipeline_time = (t_end - t_start) * 1000

# Postprocessing & Visualization (наносимо текст на кадр)
class_name = f"Class: {predicted_class.item()}, Conf: {confidence.item():.2f}"
cv2.putText(frame, class_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

print(f"Повний час обробки кадру (конвеєр): {full_pipeline_time:.2f} мс")

# Відображення результату (натисни будь-яку клавішу, щоб закрити вікно)
cv2.imshow("AI Pipeline Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()